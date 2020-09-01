# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

import os
import numpy as np
import pandas as pd
import datetime
from io import StringIO
from itertools import chain
from collections import Counter

from common.time_util import timer
from core.power_def import format_key, format_type, file_format, index_dict, \
    output_format, multi_line_header, useless_column, \
    set_format_version, get_format_version
from core.topo import PowerGraph


def f_name_conv(x):
    """ 去除字符串前后的空格和单引号。

    :param x: str.
    :return: str.
    """
    return x.strip(' \'')


def get_vl_from_vbase(v):
    """ 把浮点型电压基值转换为整数型电压等级。

    :param v: pd.Series or np.array. 浮点型电压基值列向量。
    :return: np.array. 整数型电压等级列向量。
    """
    bins = np.array([5., 30., 50., 100., 200., 300., 360., 450., 550., 650., 790., 900.])
    vls = np.array([1, 20, 35, 66, 110, 220, 330, 400, 500, 600, 750, 800, 1000])
    return vls[np.digitize(v, bins)]


def load_elem_info(file_name, etypes=[], index_col=['name']):
    """ 加载各类设备的基本信息。

    :param file_name: str. 文件路径和名称。
    :param etypes: [int]. 所需加载的设备类型。
    :param index_col: list of str. 设备列表索引列名。
    :return: pd.DataFrame. 设备信息表。
    """
    elems = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=index_col)
    if len(etypes) > 0:
        elems = elems[elems['type'].isin(etypes)]
    return elems


def load_station_info(file_name, index_col=['name']):
    """ 加载厂站的基本信息。

    :param file_name: str. 文件路径和名称。
    :param index_col: list of str. 厂站列表索引列名。
    :return: pd.DataFrame. 厂站信息表。
    """
    st_info = pd.read_table(file_name, encoding='gbk', sep=' ', index_col=index_col)
    indices = []
    for i in st_info.index:
        indices.append(i if '_' not in i else i.split('_')[0])
    st_info.index = indices
    return st_info


class Power:
    """
    处理PSASP数据文件的类。

    Attributes:
        format_key, format_type, file_format, index_dict, output_format,
        multi_line_header, useless_columns:
            dict. PSASP数据文件读写行为的配置信息。
        fmt: str. 格式模板，'on', 'off' 或用户自定义。
        data: dict of pd.DataFrame. 数据存储字典。
        station: pd.DataFrame. 厂站信息表。
    """

    def __init__(self, fmt):
        self.format_key = format_key()
        self.format_type = format_type()
        self.file_format = file_format()
        self.index_dict = index_dict()
        self.output_format = output_format()
        self.multi_line_header = multi_line_header()
        self.useless_columns = useless_column()

        self.fmt = fmt
        self.data = {}
        self.stations = None

    def get_column_and_index(self, file_name, fmt='on', ex=None):
        """ 获取文件相关的设备类型、列名、索引和文件扩展名。

        :param file_name: str. 文件路径和名称。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
                   or None. 从file_name中获取扩展名。
        :return: (etype, columns, index, ex).
        """
        if ex is None and '.' in file_name:
            ex = file_name[file_name.rindex('.') + 1:].lower()
        if fmt not in self.file_format or ex not in self.format_key:
            raise NotImplementedError("Not recognized format: (%s, %s)" % (fmt, ex))
        if ex in self.format_type['single']:
            return self.format_key[ex], self.file_format[fmt][ex], \
                   self.index_dict.get(ex, None), ex
        etypes = self.format_key[ex]
        columns = [self.file_format[fmt].get(ex + '_' + t, []) for t in etypes]
        indices = [self.index_dict.get(t, None) for t in etypes]
        return etypes, columns, indices, ex

    def get_flat_columns(self, fmt, etype, ex):
        """ 获取展开的列名，主要用于多行文件。

        :param fmt: str. 格式模板。
        :param etype: str. 设备类型。
        :param ex: str. 文件扩展名。
        :return: [columns].
        """
        if ex in self.format_type['single']:
            return self.file_format[fmt][ex]
        else:
            return list(chain(*self.file_format[fmt][ex + '_' + etype]))

    def get_flag_valid(self, df, ex):
        """ 获取有效的lp或st记录条目。

        :param df: pd.DataFrame. 数据表。
        :param ex: str. 文件扩展名。
        :return: pd.Series of bool.
        """
        if ex in self.format_key['lp']:
            return Power.get_flag(df, 'lp')
        elif ex in self.format_key['st']:
            return Power.get_flag(df, 'st')
        else:
            return pd.Series(data=True, index=df.index)

    def load_lf(self, file_name, fmt='on', ex=None, drop_useless=True):
        """ 加载 LF.L* 文件。

        :param file_name: str. 文件路径和名称。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param drop_useless: bool. 删除无用的列，节省内存开销。
        :return: pd.DataFrame or None.
        """
        if not os.path.exists(file_name):
            return None
        etype, columns, indices, ex = self.get_column_and_index(file_name, fmt, ex)
        if ex == 'l1':
            converters = {'name': f_name_conv, 'st_name': f_name_conv}
        else:
            converters = {'name': f_name_conv} if fmt == 'on' \
                else {'name': f_name_conv, 'name_ctrl': f_name_conv}
        if drop_useless and etype in self.useless_columns:
            usecols = [col for col in columns if col not in self.useless_columns[etype]]
        else:
            usecols = columns
        df = pd.read_csv(file_name, names=columns, encoding='gbk', index_col=False,
                         usecols=usecols, converters=converters)

        if ex == 'l1':
            df.index = range(1, df.shape[0] + 1)
            df['bus'] = range(1, df.shape[0] + 1)
        else:
            df['ori_order'] = range(df.shape[0])
            if ex == 'l3':
                df['ibus'] = np.abs(df['ibus'])
            if indices is not None:
                df.set_index(indices, drop=False, inplace=True)
                df.sort_index(inplace=True)
        return df

    def save_lf(self, file_name, df, fmt='on', ex=None, ori_order=True, miss='fill'):
        """ 保存 LF.L* 文件。

        :param file_name: str. 文件路径和名称。
        :param df: pd.DataFrame. 数据表。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param ori_order: bool. 采用原始顺序输出，否则按照索引排序后顺序输出。
        :param miss: str, fill or raise. 遇到缺失的列选择默认值填充或者报错。
        """
        etype, columns, _, ex = self.get_column_and_index(file_name, fmt, ex)
        if ori_order and 'ori_order' in df.columns:
            columns = columns + ['ori_order']
        miss_columns = [col for col in columns if col not in df.columns]
        ready_columns = [col for col in columns if col in df.columns]
        if miss_columns and miss == 'raise':
            raise ValueError('Save miss columns: ', miss_columns)
        valid = self.get_flag_valid(df, ex)
        sub = df.loc[valid, ready_columns].copy()
        if len(sub) > 0:
            if 'ori_order' in sub.columns:
                sub.sort_values(by='ori_order', inplace=True)
                sub.drop(columns=['ori_order'], inplace=True)
                columns.remove('ori_order')
            if miss_columns and miss == 'fill':
                self.fill_default(sub, ex, miss_columns)
                sub = sub[columns]
            if ex == 'l3':
                t3w = sub['trs_type'].isin([31, 32, 33])
                sub.loc[t3w, 'ibus'] = -sub.loc[t3w, 'ibus']
        header = '' if ex != 'lp1' else '1,' + str(datetime.datetime.now())
        formats = self.get_output_format(ex, columns=sub.columns, dtypes=sub.dtypes)
        np.savetxt(file_name, sub.values, fmt=formats,
                   newline=',\n', header=header, comments=' ', encoding='gbk')

    def load_lp(self, file_name, base=None, fmt='on', ex=None, flag='lp',
                drop_useless=True):
        """ 加载 LF.LP* 或 ST.S* 文件。

        :param file_name: str. 文件路径和名称。
        :param base: pd.DataFrame. 原始数据表，此时新读入信息会添加到原数据表的后面；
                     or None. 无原始数据表，则返回新数据表。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param flag: str. lp or st. 文件类型，lp为潮流结果，st为动态参数。
        :param drop_useless: bool. 删除无用的列。
        :return: pd.DataFrame or None.
        """
        if not os.path.exists(file_name):
            return None
        etype, columns, indices, ex = self.get_column_and_index(file_name, fmt, ex)
        skiprows = 1 if ex == 'lp1' else 0
        if drop_useless and etype in self.useless_columns:
            usecols = [col for col in columns if col not in self.useless_columns[etype]]
        else:
            usecols = columns
        df = pd.read_csv(file_name, names=columns, encoding='gbk',
                         skiprows=skiprows, usecols=usecols, index_col=False)
        df.set_index(indices, inplace=True)
        Power.set_flag(df, True, flag)
        if base is None:
            return df
        # df = df.reindex(index=base.index, fill_value=0)
        df = df.reindex(index=base.index, fill_value=np.nan)
        Power.set_flag(base, df['flag'] > 0, flag)
        columns = [col for col in df.columns if col not in base.columns]
        return pd.concat([base, df[columns]], axis=1)

    def load_st(self, file_name, base=None, fmt='on', ex=None, flag='st',
                drop_useless=True):
        """ 加载 ST.S* 文件。

        :param file_name: str. 文件路径和名称。
        :param base: pd.DataFrame. 原始数据表，此时新读入信息会添加到原数据表的后面；
                     or None. 无原始数据表，则返回新数据表。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param flag: str. lp or st. 文件类型，lp为潮流结果，st为动态参数。
        :param drop_useless: bool. 删除无用的列。
        :return: pd.DataFrame or None.
        """
        if ex is None and '.' in file_name:
            ex = file_name[file_name.rindex('.') + 1:].lower()
        if ex == 's1':
            if base is not None:
                Power.set_flag(base, True, flag)
            return None
        return self.load_lp(file_name, base=base, fmt=fmt, ex=ex, flag=flag,
                            drop_useless=drop_useless)

    def load_mlf(self, file_or_buffer, base=None, fmt='on',
                 ex=None, header_char=None, drop_useless=True):
        """ 加载多行格式的文件，包括NL4, NP4, NS4, ML4, MP4。

        :param file_or_buffer: str or StringIO. 文件路径和名称，或文件数据缓冲。
        :param base: pd.DataFrame. 原始数据表，此时新读入信息会添加到原数据表的后面。
                     or None. 无原始数据表，则返回新数据表。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param header_char: 多行格式的分隔符，ML4和MP4采用'#'；
               None. 无分隔符模式，如NL4、NP4、NS4。
        :param drop_useless: bool. 删除无用的列。
        :return: {str: pd.DataFrame}. {etype: df}.
        """
        if isinstance(file_or_buffer, StringIO):
            source = file_or_buffer
            assert ex is not None
        elif isinstance(file_or_buffer, str):
            try:
                source = open(file_or_buffer, 'r', encoding='gbk')
            except FileNotFoundError:
                return {}
        else:
            raise NotImplementedError("Unknown format. " + type(file_or_buffer))
        names, columns, indices, ex = self.get_column_and_index(file_or_buffer, fmt, ex)
        flag = 'lf'
        if ex in self.format_key['lp']:
            flag = 'lp'
        elif ex in self.format_key['st']:
            flag = 'st'
        num_lines = [len(col) for col in columns]
        columns = [list(chain(*col)) for col in columns]
        tables = dict(zip(names, [None] * len(names)))
        idx, i = -1, 0
        prompt = ''
        buffer = StringIO()
        lines = []
        for line in source:
            if header_char is not None and line[0] == header_char:
                # curr = line[:line.index(',')]
                curr = line[:2]     #sometimes lack of ','
                if prompt == curr:
                    continue
                if buffer.tell() > 0 and num_lines[idx] > 0:
                    buffer.seek(0)
                    if drop_useless and names[idx] in self.useless_columns:
                        usecols = [col for col in columns[idx]
                                   if col not in self.useless_columns[names[idx]]]
                    else:
                        usecols = columns[idx]
                    df = pd.read_csv(buffer, names=columns[idx], usecols=usecols,
                                     converters={'name': f_name_conv})
                    if indices[idx] is not None:
                        df.set_index(indices[idx], drop=False, inplace=True)
                        # df.sort_index(inplace=True)
                    Power.set_flag(df, True, flag)
                    tables[names[idx]] = df
                buffer.truncate(0)
                buffer.seek(0)
                lines.clear()
                prompt = curr
                i = 0
                idx += 1
            elif num_lines[idx] > 0:
                lines.append(line.rstrip('\r\n'))
                i += 1
                if i % num_lines[idx] == 0:
                    buffer.write(''.join(lines) + '\n')
                    lines.clear()
        if isinstance(file_or_buffer, str):
            source.close()
        if buffer.tell() > 0 and idx < len(names):
            idx = max(idx, 0)
            buffer.seek(0)
            if drop_useless and names[idx] in self.useless_columns:
                usecols = [col for col in columns[idx]
                           if col not in self.useless_columns[names[idx]]]
            else:
                usecols = columns[idx]
            df = pd.read_csv(buffer, names=columns[idx], usecols=usecols,
                             converters={'name': f_name_conv})
            if indices[idx] is not None:
                df.set_index(indices[idx], drop=False, inplace=True)
                # df.sort_index(inplace=True)
            Power.set_flag(df, True, flag)
            tables[names[idx]] = df
        buffer.close()
        if base is None:
            return tables
        for k in tables:
            if tables[k] is None or len(tables[k]) == 0:
                continue
            if k in base and base[k] is not None:
                tables[k] = tables[k].reindex(index=base[k].index, fill_value=0)
                Power.set_flag(base[k], tables[k]['flag'] > 0, flag)
                columns = [col for col in tables[k].columns if col not in base[k].columns]
                base[k] = pd.concat([base[k], tables[k][columns]], axis=1)
            else:
                base[k] = tables[k]
        return base

    def save_mlf(self, file_name, data, fmt='on', ex=None, miss='fill'):
        """ 保存多行格式的文件，包括NL4, NP4, NS4, ML4, MP4。

        :param file_name: str. 文件路径和名称。
        :param data: {str: pd.DataFrame}. {etype: df}. 数据字典。
        :param fmt: str. 格式模板。
        :param ex: str. 文件扩展名。
        :param miss: str, fill or raise. 缺失列处理方式。
        """
        etypes, columns, _, ex = self.get_column_and_index(file_name, fmt, ex)
        headers = self.multi_line_header[ex]
        with open(file_name, 'w') as fp:
            for i, e in enumerate(etypes):
                if headers is not None and i < len(headers):
                    fp.write('\n'.join(headers[i]) + '\n')
                if not columns[i] or e not in data \
                        or data[e] is None or len(data[e]) == 0:
                    print('Not recognized format (%s, %s) or empty data' % (ex, e))
                    continue
                cols = list(chain(*columns[i]))
                miss_cols = [col for col in cols if col not in data[e].columns]
                ready_cols = [col for col in cols if col in data[e].columns]
                if miss_cols and miss == 'raise':
                    raise ValueError('Save miss columns: ', miss_cols)
                valid = self.get_flag_valid(data[e], ex)
                sub = data[e].loc[valid, ready_cols].copy()
                if miss_cols and miss == 'fill' and len(sub) > 0:
                    self.fill_default(sub, ex, columns=miss_cols)
                    sub = sub[cols]
                out_fmt = [self.get_output_format(ex, columns=col, dtypes=sub.dtypes[col])
                           for col in columns[i]]
                out_fmt = ',\n'.join(out_fmt) + ',\n'
                for _, *values in sub.itertuples():
                    fp.write(out_fmt % tuple(values))

    def get_output_format(self, ex, columns, dtypes):
        """ 获取输出文件的格式字符串。

        :param ex: str. 文件扩展名。
        :param columns: [str]. 列名列表。
        :param dtypes: pd.Series. 数据表每列的数据类型。
        :return: str.
        """
        default = self.output_format['default']
        if ex not in self.output_format:
            formats = [default[dtypes[col].kind][1] for col in columns]
        else:
            formats = [self.output_format[ex].get(col, default[dtypes[col].kind])[1]
                       for col in columns]
        return ','.join(formats)

    def check_columns(self, df, etype, columns=None, ex=None, fmt='on'):
        """ 检查缺失列。

        :param df: pd.DataFrame. 数据表。
        :param etype: str. 设备类型。
        :param columns: [str]. 列名列表。
        :param ex: str. 文件扩展名。
        :param fmt: str. 格式模板。
        :return: [str]. 缺失列的列表。
        """
        if not columns:
            if isinstance(self.format_key[ex], list):
                columns = list(chain(self.file_format[fmt][ex + '_' + etype]))
            else:
                columns = self.file_format[fmt][ex]
        return [col for col in df.columns if col not in columns]

    def fill_default(self, df, ex, columns):
        """ 按默认值进行填充。

        :param df: pd.DataFrame. 数据表。
        :param ex: str. 文件扩展名。
        :param columns: [str]. 目标列名的列表。
        """
        for col in columns:
            value = self.output_format[ex][col][0]
            if value is None:
                raise ValueError('Column [%s] has no default value' % col)
            if col not in df.columns:
                df[col] = value
            else:
                df[col].fillna(value, inplace=True)

    @staticmethod
    def set_flag(df, valid, flag):
        """ 设置lp或st标志位。

        :param df: pd.DataFrame. 数据表。
        :param valid: bool or [bool]. 有效标志位。
        :param flag: str, lp or st. 标志类型。
        """
        if flag == 'lp':
            bit = 1
        elif flag == 'st':
            bit = 2
        else:
            return
        bits = valid * bit
        if 'flag' not in df.columns:
            df['flag'] = bits
        else:
            df['flag'] &= ~bit
            df['flag'] |= bits

    @staticmethod
    def get_flag(df, flag):
        """ 获取lp或st标志位。

        :param df: pd.DataFrame. 数据表。
        :param flag: str, lp or st. 标志类型。
        :return: pd.Series. 有效标志位列表。
        """
        if 'flag' not in df.columns:
            return pd.Series(data=False, index=df.index, name='flag')
        bit = 1 if flag == 'lp' else 2
        return df['flag'] & bit > 0

    def drop_data(self, fmt, flag, etypes=None):
        """ 删除lp或st的相关数据。

        :param etypes: [str]. 设备类型列表。
                       or None. 所有设备。
        :param fmt: str. 格式模板。
        :param flag: str, lp or st. 标志类型。
        """
        if not etypes:
            etypes = self.data.keys()
        for etype in etypes:
            if flag == 'lp':
                ex1 = self.format_key[etype][1]
            elif flag == 'st':
                ex1 = self.format_key[etype][2]
            flags = Power.get_flag(self.data[etype], flag)
            if ex1 is None or not np.any(flags):
                continue
            columns1 = self.get_flat_columns(fmt, etype, ex1)
            columns0 = self.get_flat_columns(fmt, etype, self.format_key[etype][0])
            drops = [col for col in columns1
                     if col not in columns0 and col in self.data[etype].columns]
            self.data[etype].drop(columns=drops, inplace=True)
            Power.set_flag(self.data[etype], False, flag)

    def describe(self):
        """
        统计lf、lp和st的有效记录数据。
        """
        for name, df in self.data.items():
            n_lf = df.shape[0]
            n_lp = Power.get_flag(df, 'lp').sum()
            n_st = Power.get_flag(df, 'st').sum()
            print('[%s]: n_lf=%d, n_lp=%d, n_st=%d' % (name, n_lf, n_lp, n_st))

    def load_power(self, path, fmt='on', lf=True, lp=True, st=True,
                   station=True, shorten=True):
        """ 加载数据目录。

        :param path: str. 数据目录。
        :param fmt: str. 格式模板。
        :param lf: bool. 是否加载lf文件。
        :param lp: bool. 是否加载lp文件。
        :param st: bool. 是否加载st文件。
        :param station: bool. 是否生成厂站信息。
        :param shorten: bool. 采用短格式，用于节省内存开销。
        """
        if lf:
            self.data['bus'] = self.load_lf(path + '/LF.L1', fmt)
            self.data['bus']['vl'] = get_vl_from_vbase(self.data['bus']['vbase'])
            self.data['acline'] = self.load_lf(path + '/LF.L2', fmt)
            self.data['transformer'] = self.load_lf(path + '/LF.L3', fmt)
            self.data['generator'] = self.load_lf(path + '/LF.L5', fmt)
            self.data['load'] = self.load_lf(path + '/LF.L6', fmt)
            self.data.update(self.load_mlf(path + '/LF.NL4', fmt=fmt))
            self.data.update(self.load_mlf(path + '/LF.ML4', fmt=fmt, header_char='#'))
            if 'dcbus' in self.data:
                self.data['dcbus']['name'] =\
                    self.data['bus'].loc[self.data['dcbus']['bus'], 'name']
        if lp:
            self.data['bus'] = self.load_lp(path + '/LF.LP1', self.data['bus'], fmt)
            self.data['acline'] = self.load_lp(path + '/LF.LP2',
                                               self.data['acline'], fmt)
            self.data['transformer'] = self.load_lp(path + '/LF.LP3',
                                                    self.data['transformer'], fmt)
            self.data['generator'] = self.load_lp(path + '/LF.LP5',
                                                  self.data['generator'], fmt)
            self.data['load'] = self.load_lp(path + '/LF.LP6', self.data['load'], fmt)
            self.data.update(self.load_mlf(path + '/LF.NP4', self.data, fmt))
            self.data.update(self.load_mlf(path + '/LF.MP4', self.data, fmt,
                                           header_char='#'))
        if st:
            # self.data['bus'] = self.load_st(path+'/ST.S1', self.data['bus'], fmt)
            Power.set_flag(self.data['bus'], True, 'st')
            self.data['acline'] = self.load_st(path + '/ST.S2', self.data['acline'], fmt)
            self.data['transformer'] = self.load_st(path + '/ST.S3',
                                                    self.data['transformer'], fmt)
            self.data['generator'] = self.load_st(path + '/ST.S5',
                                                  self.data['generator'], fmt)
            self.data['load'] = self.load_st(path + '/ST.S6', self.data['load'], fmt)
            self.data.update(self.load_mlf(path + '/ST.NS4', self.data, fmt))
            if 'vsc' in self.data:
                self.data['vsc'] = self.load_st(path + '/ST.MS4', self.data['vsc'], fmt)
        if shorten:
            self.shorten_storage(fmt=fmt)
        if station:
            self.generate_station_info()
        # self.describe()

    def shorten_storage(self, etypes=None, fmt='on'):
        """ 采用短格式存储，整数型采用int32，浮点型采用float32。

        :param etypes: str. 设备类型。
        :param fmt: str. 格式模板。
        """
        if not etypes:
            etypes = self.data.keys()
        else:
            etypes = [e for e in etypes if e in self.data]
        for e in etypes:
            if not isinstance(self.data[e], pd.DataFrame):
                continue
            dtypes = self.data[e].dtypes
            ex = self.format_key[e][1]
            lp_columns = self.get_flat_columns(fmt, e, ex)
            columns = [col for col in self.data[e].columns if dtypes[col].kind == 'i']
            if columns:
                self.data[e][columns] = self.data[e][columns].astype('int32')
            columns = [col for col in self.data[e].columns
                       if dtypes[col].kind == 'f' and col not in lp_columns]
            if columns:
                self.data[e][columns] = self.data[e][columns].astype('float32')

    def generate_mdc_version_outline(self, fmt='on'):
        """ 生成mdc的version和outline信息。

        :param fmt: str. 格式模板。
        """
        self.data['version'] = pd.DataFrame([get_format_version('mdc')],
                                            columns=['version'])
        Power.set_flag(self.data['version'], True, 'lp')
        columns = self.file_format[fmt]['ml4_outline'] \
                  + self.file_format[fmt]['mp4_outline']
        columns = list(chain(*columns))
        values = [0] * 12
        values[5] = 1  # method
        if 'dcbus' in self.data and self.data['dcbus'] is not None:
            values[0] = self.data['dcbus'].shape[0]
            values[7] = np.sum(Power.get_flag(self.data['dcbus'], 'lp'))
        if 'vsc' in self.data and self.data['vsc'] is not None:
            values[1] = self.data['vsc'].shape[0]
            values[8] = np.sum(Power.get_flag(self.data['vsc'], 'lp'))
        if 'lcc' in self.data and self.data['lcc'] is not None:
            values[2] = self.data['lcc'].shape[0]
            values[9] = np.sum(Power.get_flag(self.data['lcc'], 'lp'))
        if 'mdcline' in self.data and self.data['mdcline'] is not None:
            values[3] = self.data['mdcline'].shape[0]
            values[10] = np.sum(Power.get_flag(self.data['mdcline'], 'lp'))
        if 'dcdc' in self.data and self.data['dcdc'] is not None:
            values[4] = self.data['dcdc'].shape[0]
            values[11] = np.sum(Power.get_flag(self.data['dcdc'], 'lp'))
        if 'dcsys' in self.data and self.data['dcsys'] is not None:
            values[6] = self.data['dcsys'].shape[0]
        self.data['outline'] = pd.DataFrame([values], columns=columns, index=[0])
        Power.set_flag(self.data['outline'], True, 'lp')

    def generate_island_info(self):
        """
        生成分岛信息。
        """
        graph = PowerGraph(self, graph_type='multi', node_type='bus')
        islands = graph.get_islands(10)
        self.data['bus']['island'] = islands

    def get_largest_island(self):
        """ 获取母线数量最多的电气岛。

        :return: int. 最大岛的岛号。
        """
        if 'island' not in self.data['bus'].columns:
            self.generate_island_info()
        counts = self.data['bus']['island'].value_counts()
        if counts.shape[0] == 0:
            raise ValueError("No island data.")
        return counts.index[0]

    def generate_station_info(self):
        """
        生成厂站信息。
        """
        if 'island' not in self.data['bus'].columns:
            self.generate_island_info()
        names = ['%s_%d' % (n, i) for idx, n, i in
                 self.data['bus'][['st_name', 'island']].itertuples()]
        self.stations = pd.DataFrame(data=list(set(names)), columns=['unique_name'])
        self.stations['island'] = \
            [int(name.split('_')[-1]) for name in self.stations['unique_name']]
        self.stations['name'] = \
            [name[:name.rindex('_')] for name in self.stations['unique_name']]
        name_idx = pd.DataFrame(data=range(self.stations.shape[0]),
                                index=self.stations['unique_name'])
        self.data['bus']['st_no'] = name_idx.loc[names].values
        self.stations = self.stations[self.stations['island'] >= 0]
        self.stations.loc[-1] = ['Unplugged_-1', -1, 'Unplugged']
        self.data['bus'].loc[self.data['bus']['island'] < 0, 'st_no'] = -1
        self.stations['vl'] = self.data['bus'].groupby('st_no', sort=False)\
            .agg({'vl':'max'})
        self.data['acline']['st_i'] = \
            self.data['bus'].loc[self.data['acline'].ibus, 'st_no'].values
        self.data['acline']['st_j'] = \
            self.data['bus'].loc[self.data['acline'].jbus, 'st_no'].values
        self.data['transformer']['st_no'] = \
            self.data['bus'].loc[self.data['transformer'].ibus, 'st_no'].values
        self.data['generator']['st_no'] = \
            self.data['bus'].loc[self.data['generator'].bus, 'st_no'].values
        self.data['load']['st_no'] = \
            self.data['bus'].loc[self.data['load'].bus, 'st_no'].values
        self.data['station'] = self.stations

    def extend_lp_info(self, real_only=False):
        """ 扩展潮流结果的存储方式，便于查询，但存在重复数据，更加占用空间。

        :param real_only: bool. 只考虑真实设备。
        :return:
        """
        if 'generator' in self.data:
            self.data['generator']['v'] = \
                self.data['bus'].loc[self.data['generator']['bus'], 'v']
            not_valid = ~Power.get_flag(self.data['generator'], 'lp')
            self.data['generator'].loc[not_valid, 'v'] = np.nan
        if 'load' in self.data:
            self.data['load']['v'] = \
                self.data['bus'].loc[self.data['load']['bus'], 'v'].values
            not_valid = ~Power.get_flag(self.data['load'], 'lp')
            self.data['load'].loc[not_valid, 'v'] = np.nan
        if 'station' in self.data:
            generators = self.data['generator']
            valid = generators['mark'] == 1
            if real_only:
                valid &= ~generators['name'].str.contains('@')
            generators = generators[valid]
            self.data['station'][['pg', 'qg']] = generators.groupby('st_no', sort=False) \
                .agg({'p': 'sum', 'q': 'sum'})
            loads = self.data['load']
            valid = loads['mark'] == 1
            if real_only:
                valid &= ~loads['name'].str.contains('@')
            loads = loads[valid]
            self.data['station'][['pl', 'ql']] = loads.groupby('st_no', sort=False) \
                .agg({'p': 'sum', 'q': 'sum'})

    def save_power(self, path, fmt='on', lf=True, lp=True, st=True, miss='fill'):
        """ 保存数据目录。

        :param path: str. 数据目录。
        :param fmt: str. 格式模板。
        :param lf: bool. 是否保存lf文件。
        :param lp: bool. 是否保存lp文件。
        :param st: bool. 是否保存st文件。
        :param miss: str, fill or raise. 缺失列处理方式。
        """
        if fmt != self.fmt:
            print("Format mismatch: load is %s, save is %s" % (self.fmt, fmt))
        if not os.path.exists(path):
            os.mkdir(path)
        if lf:
            self.save_lf(path + '/LF.L1', self.data['bus'], fmt=fmt,
                         ori_order=False, miss=miss)
            self.save_lf(path + '/LF.L2', self.data['acline'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.L3', self.data['transformer'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.L5', self.data['generator'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.L6', self.data['load'], fmt=fmt, miss=miss)
            if 'dcline' in self.data:
                self.save_mlf(path + '/LF.NL4', self.data, fmt=fmt, miss=miss)
            if 'dcbus' in self.data:
                self.save_mlf(path + '/LF.ML4', self.data, fmt=fmt, miss=miss)
        if lp:
            self.save_lf(path + '/LF.LP1', self.data['bus'], fmt=fmt,
                         ori_order=False, miss=miss)
            self.save_lf(path + '/LF.LP2', self.data['acline'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP3', self.data['transformer'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP5', self.data['generator'], fmt=fmt, miss=miss)
            self.save_lf(path + '/LF.LP6', self.data['load'], fmt=fmt, miss=miss)
            if 'dcline' in self.data:
                self.save_mlf(path + '/LF.NP4', self.data, fmt=fmt, miss=miss)
            if 'dcbus' in self.data:
                self.save_mlf(path + '/LF.MP4', self.data, fmt=fmt, miss=miss)
        if st:
            self.save_lf(path + '/ST.S1', self.data['bus'], fmt=fmt,
                         ori_order=False, miss=miss)
            self.save_lf(path + '/ST.S2', self.data['acline'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S3', self.data['transformer'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S5', self.data['generator'], fmt=fmt, miss=miss)
            self.save_lf(path + '/ST.S6', self.data['load'], fmt=fmt, miss=miss)
            if 'dcline' in self.data:
                self.save_mlf(path + '/ST.NS4', self.data, fmt=fmt, miss=miss)
            if 'vsc' in self.data:
                self.save_lf(path + '/ST.MS4', self.data['vsc'], fmt=fmt, miss=miss)

    @staticmethod
    def statistic_acline(aclines):
        """ 统计交流线表数据。

        :param aclines: pd.DataFrame. 交流线表。
        """
        valid_on = aclines['mark'] > 0
        valid_pc = aclines['ibus'] == aclines['jbus']
        # valid_r0 = aclines['r'] == 0.
        valid_x0 = aclines['x'].abs() <= 1e-4
        valid_xn = aclines['x'] < -1e-4
        valid_xp = aclines['x'] > 1e-4
        count_ac = aclines.shape[0]
        count_ac_on = np.sum(valid_on)
        count_pc = np.sum(valid_pc & valid_xn)
        count_pc_on = np.sum(valid_pc & valid_xn & valid_on)
        count_pr = np.sum(valid_pc & valid_xp)
        count_pr_on = np.sum(valid_pc & valid_xp & valid_on)
        count_sr = np.sum(~valid_pc & valid_xn)
        count_sr_on = np.sum(~valid_pc & valid_xn & valid_on)
        count_bs = np.sum(valid_x0)
        count_bs_on = np.sum(valid_x0 & valid_on)
        print('acline=%d, on=%d' % (count_ac, count_ac_on))
        print('pc=%d, on=%d' % (count_pc, count_pc_on))
        print('pr=%d, on=%d' % (count_pr, count_pr_on))
        print('sr=%d, on=%d' % (count_sr, count_sr_on))
        print('bs=%d, on=%d' % (count_bs, count_bs_on))


if __name__ == '__main__':
    # set_format_version({'mdc': 2.3})
    path = 'D:/PSA_src/psa/localdata/0913/data'
    fmt = 'on'
    path = 'D:/PSASP_Pro/2020国调年度/冬低731'
    fmt = 'off'
    power = Power(fmt)
    with timer('Load power'):
        power.load_power(path, fmt=fmt, lp=False, st=True, station=False)
    with timer('Save power'):
        power.save_power(path + '/out', lp=False, st=True, fmt=fmt)
