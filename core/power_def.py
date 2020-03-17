# coding: gbk
"""
@author: sdy
@email: sdy@epri.sgcc.com.cn
"""

from itertools import chain
from copy import deepcopy


format_versions = {'mdc': 2.4}


def set_format_version(versions):
    global format_versions, slf, mlf
    format_versions.update(versions)
    # slf = multi_line_format()
    mlf = multi_line_format()


def get_format_version(k):
    global format_versions
    return format_versions.get(k, 0.0)


def format_key():
    return dict(
        # type key infos
        types=['bus', 'acline', 'transformer', 'generator', 'load', 'dcline',
               'version', 'outline', 'dcsys', 'dcbus', 'vsc', 'lcc', 'mdcline', 'dcdc'],
        bus=('l1', 'lp1', 's1'),
        acline=('l2', 'lp2', 's2'),
        transformer=('l3', 'lp3', 's3'),
        generator=('l5', 'lp5', 's5'),
        load=('l6', 'lp6', 's6'),
        dcline=('nl4', 'np4', 'ns4'),
        version=('ml4', 'mp4', None),
        outline=('ml4', 'mp4', None),
        dcsys=(None, 'mp4', None),
        dcbus=('ml4', 'mp4', None),
        vsc=('ml4', 'mp4', 'ms4'),
        lcc=('ml4', 'mp4', None),
        mdcline=('ml4', 'mp4', None),
        dcdc=('ml4', 'mp4', None),
        # file key infos
        lf=('l1', 'l2', 'l3', 'l5', 'l6', 'nl4', 'ml4'),
        lp=('lp1', 'lp2', 'lp3', 'lp5', 'lp6', 'np4', 'mp4'),
        st=('s1', 's2', 's3', 's5', 's6', 'ns4', 'ms4'),
        l1='bus', lp1='bus', s1='bus',
        l2='acline', lp2='acline', s2='acline',
        l3='transformer', lp3='transformer', s3='transformer',
        l5='generator', lp5='generator', s5='generator',
        l6='load', lp6='load', s6='load',
        nl4=['dcline'], np4=['dcline'], ns4=['dcline'],
        ml4=['version', 'outline', 'dcbus', 'vsc', 'lcc', 'mdcline', 'dcdc'],
        mp4=['version', 'outline', 'dcsys', 'dcbus', 'vsc', 'lcc', 'mdcline', 'dcdc'],
        ms4='vsc'
    )


def format_type():
    return dict(
        single=['l1', 'l2', 'l3', 'l5', 'l6',
                'lp1', 'lp2', 'lp3', 'lp5', 'lp6',
                's1', 's2', 's3', 's5', 's6', 'ms4'],
        multi=['nl4', 'np4', 'ns4', 'ml4', 'mp4']
    )


def single_line_format():
    return dict(
        l1=['name', 'vbase', 'area', 'vmax', 'vmin',
            'cb1', 'cb3', 'st_name', 'island'],
        l2=['mark', 'ibus', 'jbus', 'no', 'r', 'x', 'b', 'area', 'type',
            'pl', 'cb', 'cl', 'vqp', 'lc', 'lim', 'name_ctrl', 'type_ctrl', 'name'],
        l3=['mark', 'ibus', 'jbus', 'no', 'r', 'x', 'tk', 'rm', 'xm',
            'area', 'type', 'pl', 'tp', 'cb', 'cl', 'vqp', 'theta',
            'tc', 'lim', 'id', 'j', 'trs_type', 'name_ctrl', 'type_ctrl', 'name',
            'uni', 'unj', 'tapmax', 'tapmin', 'tapun', 'du', 'tappos'],
        l5=['mark', 'bus', 'type', 'p0', 'q0', 'v0', 'theta0',
            'qmax', 'qmin', 'pmax', 'pmin', 'pl', 'cb', 'cl', 'vqp',
            'k', 'name_ctrl', 'type_ctrl', 'name'],
        l6=['mark', 'bus', 'no', 'type', 'p0', 'q0', 'v0', 'theta0',
            'qmax', 'qmin', 'pmax', 'pmin', 'pl', 'cb', 'cl',
            'vqp', 'name_ctrl', 'type_ctrl', 'tmp', 'name'],
        lp1=['bus', 'v', 'theta', 'name'],
        lp2=['ibus', 'jbus', 'no', 'pi', 'qi', 'pj', 'qj', 'qci', 'qcj',
             'name'],
        lp3=['ibus', 'jbus', 'no', 'pi', 'qi', 'pj', 'qj', 'pm', 'qm', 'name'],
        lp5=['bus', 'p', 'q', 'no', 'name'],
        lp6=['bus', 'no', 'p', 'q', 'name'],
        s1=['name'],
        s2=['mark0', 'ibus', 'jbus', 'no', 'r0', 'x0', 'b0', 'name'],
        s3=['mark0', 'ibus0', 'jbus0', 'no', 'r0', 'x0', 'tk0', 'trs_type',
            'gm0', 'bm0', 'name'],
        s5=['mark', 'bus', 'tg', 'lg', 'tvr', 'lvr', 'tgo', 'lgo', 'tpss', 'lpss',
            'xdp', 'xdpp', 'x2', 'tj', 'sh', 'ph', 'name'],
        s6=['mark', 'bus', 'no', 'tp', 'lp', 'k', 'name'],
        ms4=['mark', 'acbus', 'dch_bus', 'no', 'mode_ctrl', 'par_no', 'name']
    )


slf = single_line_format()


def multi_line_format():
    dcline = dict(
        nl4_dcline=[['mark', 'ibus', 'jbus', 'no', 'area', 'name'],
                    ['rpi', 'lpi', 'rpj', 'lpj', 'rl',
                     'll', 'rei', 'rej', 'lsi', 'lsj'],
                    ['vdn'],
                    ['vhi', 'vli', 'bi', 'sti', 'rti', 'xti', 'vtimax', 'vtimin',
                     'ntapi', 'r0i_pu', 'x0i_pu'],
                    ['vhj', 'vlj', 'bj', 'stj', 'rtj', 'xtj', 'vtjmax', 'vtjmin',
                     'ntapj', 'r0j_pu', 'x0j_pu'],
                    ['op', 'qci', 'qcj'],
                    ['pd1', 'vd1', 'a1min', 'a10', 'gama1min', 'gama10'],
                    ['pd2', 'vd2', 'a2min', 'a20', 'gama2min', 'gama20']],
        np4_dcline=[['ibus', 'jbus', 'no', 'area', 'name'],
                    ['op'],
                    ['id10_pu', 'id20_pu', 'vaci_pu', 'vacj_pu'],
                    ['pd1i_pu', 'qd1i_pu', 'vd10i_pu', 'tk1i_pu', 'a10r', 'ttk1i'],
                    ['pd1j_pu', 'qd1j_pu', 'vd10j_pu', 'tk1j_pu', 'ttk1j'],
                    ['pd2i_pu', 'qd2i_pu', 'vd20i_pu', 'tk2i_pu', 'a20r', 'ttk2i'],
                    ['pd2j_pu', 'qd2j_pu', 'vd20j_pu', 'tk2j_pu', 'ttk2j'],
                    ['vbi', 'vbj', 'vdb', 'idb', 'zdb', 'ldb', 'tkbi', 'tkbj'],
                    ['rpi_pu', 'lpi_pu', 'rpj_pu', 'lpj_pu', 'rl_pu', 'll_pu',
                     'rei_pu', 'rej_pu', 'lsi_pu', 'lsj_pu'],
                    ['xci_pu', 'xcj_pu', 'tkimax_pu', 'tkimin_pu', 'tkjmax_pu',
                     'tkjmin_pu', 'qcip_pu', 'qcjp_pu']],
        ns4_dcline=[['mark', 'ibus', 'jbus', 'no', 'dcm', 'name'],
                    ['reg_type', 'reg_fon', 'reg_par', 'reg_set', 'amax', 'amin'],
                    ['ireg_fon', 'ireg_par', 'ireg_set', 'vreg_fon', 'vreg_par',
                     'vreg_set', 'greg_fon', 'greg_par', 'greg_set',
                     'bmax', 'bmin', 'im'],
                    ['v_low', 'idmax', 'idmin', 'dki', 'dkj',
                     'dti', 'dtj', 'v_relay', 't_relay'],
                    ['type_fdc', 'k_fdc', 'ts_fdc', 'te_fdc', 'tdo', 'tda', 'tdb', 'tdc',
                     'n_rs', 'v_rs']]
    )
    mdc_v23 = dict(
        ml4_version=[['version']],
        ml4_outline=[['bus_num', 'vsc_num', 'lcc_num',
                      'line_num', 'dcdc_num', 'method']],
        ml4_dcbus=[['bus', 'type', 'dtype', 'dr', 'dx', 'lr', 'lx']],
        ml4_vsc=[['mark', 'acbus', 'dch_bus', 'dcl_bus', 'no', 'ac_kv',
                  'ac_mva', 'rc', 'lc', 'dp_num', 'mc', 'loss_rate', 'name'],
                 ['pole_num', 'mw', 'mvar', 'p_ctrl', 'ctrl_method',
                  'ref_kv', 'ref_angle', 'balance', 'link', 'dc_ref_kv',
                  'filter_mvar', 'max_dc_ka', 'min_mod', 'max_mod']],
        ml4_lcc=[['mark', 'acbus', 'dch_bus', 'dcl_bus',
                  'no', 'of', 'lccid', 'confr', 'name'],
                 ['bi', 'sti', 'rti', 'xti', 'vhi', 'vli',
                  'vti0', 'vtimax', 'vtimin', 'ntap', 'rt0', 'xt0'],
                 ['iri', 'icv', 'ics', 'iset', 'pset', 'vset', 'aset', 'amin']],
        ml4_mdcline=[['mark', 'ibus', 'jbus', 'no', 'r', 'x', 'c',
                      'cmax', 'ratei', 'name']],
        ml4_dcdc=[['mark', 'ibus', 'jbus', 'no', 'pset', 'g', 'r', 'name']],
        mp4_version=[['version']],
        mp4_outline=[['dcsys_num', 'bus_on_num', 'vsc_on_num', 'lcc_on_num',
                      'line_on_num', 'dcdc_on_num']],
        mp4_dcsys=[['id', 'dcbus_num', 'vsc_num,',
                    'line_num', 'lcc_num', 'lcc_type']],
        mp4_dcbus=[['bus', 'id_sys', 'vdc', 'name', 'type',
                    'gi', 'ground', 're', 'le', 'lre', 'lle']],
        mp4_vsc=[['acbus', 'dch_bus', 'dcl_bus', 'no', 'rbus',
                  'id_sys', 'name', 'ibus0', 'ibus3'],
                 ['p_sys', 'q_sys', 'qf0', 'ps0', 'qs0', 'pc0', 'qc0', 'u_sys',
                  'delta_u_sys', 'us0', 'delta_us0', 'uc0', 'delta_uc0',
                  'p3', 'q3', 'u3', 'delta_u3', 'u0', 'delta_u0'],
                 ['isd0', 'isq0', 'udc_h', 'udc_l', 'idc', 'pdc'],
                 ['rt_pu', 'lt_pu', 'tk', 'rc_pu',
                  'lc_pu', 'vbase', 'sbase', 'zbase'],
                 ['udbase', 'idbase', 'nvol', 'cd', 'kcost', 'npol',
                  'flag_pq', 'flag_bs', 'udn', 'ccmax', 'umin', 'umax']],
        mp4_lcc=[['acbus', 'dch_bus', 'dcl_bus', 'no', 'id_sys', 'name'],
                 ['p_sys', 'q_sys', 'qf0', 'u_sys', 'delta_u_sys',
                  'udc_h', 'udc_l', 'idc', 'pdc'],
                 ['tk0', 'ntap0', 'a0', 'bi', 'amin'],
                 ['id', 'flag_rihl', 'rt_pu', 'lt_pu', 'kv_max', 'kv_min',
                  'basekv_ac1', 'basekv_ac2', 'confr', 'rt0_pu', 'xt0_pu'],
                 ['ccw', 'iset_lp', 'pset_lp', 'vset', 'aset']],
        mp4_mdcline=[['ibus', 'jbus', 'no', 'id_sys', 'pi', 'pj', 'idc',
                      'rl', 'll', 'cl', 'ratei', 'type', 'name']],
        mp4_dcdc=[['ibus', 'jbus', 'no', 'id_sys', 'pr', 'pl', 'name']]
    )
    mdc_v24 = deepcopy(mdc_v23)
    mdc_v24['mp4_lcc'][3] = mdc_v24['mp4_lcc'][3] + ['ntap']
    mdc = mdc_v24 if get_format_version('mdc') == 2.4 else mdc_v23
    return dict(
        dcline=dcline,
        mdc=mdc
    )


mlf = multi_line_format()


def multi_line_header():
    return dict(
        nl4=None,
        np4=None,
        ns4=None,
        ml4=[["#0,�汾��"],
             ["#1,ֱ��ĸ��,VSC����վ,LCC����վ,ֱ����·, DC/DC�任��������,��ֱ�����������ʽ(0˳�򷨣�1�������淨),"],
             ["#2,ֱ��ĸ�ߺ�,ֱ��ĸ������(1:����ĸ��;2:����ĸ��;3:�ӵص�;4:�����ӵ�),�ӵط�ʽ(0:���ӵ�,1:�ӵ�),��ص�Ч����(ŷ),��ص�Ч���(��),�ӵ����ߵ���(ŷ),�ӵ����ߵ��(��),"],
             [
                 "#3,VSC����վ��Ч(0��Ч;1��Ч),����ĸ�ߺ�(���߲�),��ѹ��ֱ��ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,���,�������׼��ѹ(kV),�����(MVA),Rc(ŷ),Lc(��),��ƽ��,��ģ�����(΢��),���ϵ��,VSC����,",
                 "#3,����,�����й�(MW����Ϊ��),�����޹�(MVar),���ʿ��Ƶ��־(0ԭ��;1����),���Ʒ�ʽ(1PQ;-1PV;0ƽ��),������ѹ��ֵ(kV),������ѹ���(��),ƽ��վ��־(0��ƽ��վ;1ƽ��վ),���ӷ�ʽ(0����;1����),ֱ��������ѹ(kV),�˲�������(MVar),��������(kA),���Ʊ�����,���Ʊȵ�ѹ����,"],
             ["#4,LCC����վ��Ч(0��Ч;1��Ч),����ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,���,�˲�������(MVar),����������id,ƽ��(mH),LCC����,",
              "#4,ÿ����������,�����ѹ������������(MVA),ͭ�����ֵ(Ohm),©��ֵ(%),�������ߵ�ѹ(kV),������ڶ�ߵ�ѹ(kV),��ʼ��ͷ�ߵ�ѹ(kV),��߳�ͷ�ߵ�ѹ(kV),��ͳ�ͷ�ߵ�ѹ(kV),��ͷ����,���������(p.u.),������翹(p.u.),",
              "#4,����������(0������;1�����),����ģʽ(0����ȼ�ǵ����;1����ȵ���ȼ��),ֱ�����Ʒ�ʽ(1��ֱ������;2��ֱ������;3��ֱ����ѹ;4����ȼ��),����ֱ������(kA),����ֱ������(MW),����ֱ����ѹ(kV), ��ȼ��/Ϩ���ǳ�ʼֵ, ��ȼ��/Ϩ������Сֵ,"],
             ["#5,ֱ������Ч(0��Ч;1��Ч),i��ĸ�ߺ�,j��ĸ�ߺ�,���,����(ŷ),���(��),����(΢��),��������(kA), �����(kA),ֱ��������,"],
             ["#6,DC/DC�任����Ч(0��Ч;1��Ч),R��ĸ�ߺ�,L��ĸ�ߺ�,���,R����ƹ���(MW������Ϊ��),�絼(΢������),����(ŷ),DC/DCԪ������,"]],
        mp4=[["#0,�汾��"],
             ["#1,ֱ��ϵͳ,ֱ��ĸ��,VSC����վ,LCC����վ,ֱ����,DC/DC�任��������,"],
             ["#2,ֱ��ϵͳ���,ֱ��ϵͳ��ֱ��ĸ������,VSC����վ����,ֱ��������,LCC����վ����,����,"],
             [
                 "#3,ֱ��ĸ�ߺ�,����ֱ��ϵͳ��,��ѹ(kV),ĸ����,ĸ������(1:����ĸ��;2:����ĸ��;3:�ӵص�;4:�����ӵ�),�ӵط�ʽ(0:���ӵ�,1:ֱ�ӽӵ�),�ӵص���(kA),��ص�Ч����(ŷ),��ص�Ч���(��),�ӵ����ߵ���(ŷ), �ӵ����ߵ��(��),"],
             ["#4,VSC����վ������ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,���,ԭ�ཻ߲��ĸ�ߺ�,����ֱ��ϵͳ��,VSC����,���ĵ�ĸ�ߺ�,����������ĸ�ߺ�,",
              "#4,ԭ�߲��й�(p.u.��ͬ),�޹�,�˲����޹�,���߲��й�,�޹�,�����й�,�޹�,ԭ�߲��ѹ��ֵ,���(��),���߲��ѹ��ֵ,���(��),�����ѹ��ֵ,���,�������й�,�޹�,��ѹ��ֵ,���(��),���ĵ��ѹ��ֵ,���(��),",
              "#4,d�����(p.u.),q�����(p.u.),��ѹ��ֱ����ѹ(kV),��ѹ��ֱ����ѹ(kV),ֱ����ע�����(kA),ֱ����ע���й�(����Ϊ��MW),",
              "#4,������ѹ������(p.u.),�翹(p.u.),���(p.u.),�����(p.u.),��翹(p.u.),�������׼��ѹ(kV),�����(MVA),��迹(ŷ),",
              "#4,ֱ�����׼��ѹ(kV),ֱ�����׼����(kA),��ƽ��,��ģ�����(΢��),���ϵ��,����,���ʿ��Ƶ��־,ƽ��վ��־,ֱ��������ѹ(kV),��������(kA),���Ʊ�����,���Ʊȵ�ѹ����,"],
             ["#5,LCC����վ������ĸ�ߺ�, ��ѹ��ֱ��ĸ�ߺ�,��ѹ��ֱ��ĸ�ߺ�,���,����ֱ��ϵͳ��,LCC����,",
              "#5,�������й�(p.u.��ͬ),�޹�,�˲����޹�,��ѹ��ֵ,���(��), ��ѹ��ֱ����ѹ(kV),��ѹ��ֱ����ѹ(kV),ֱ����ע�����(kA),ֱ����ע���й�(����Ϊ��MW),",
              "#5,�����ѹ��ʵ�ʱ��,��ͷλ��(kV),ʵ�ʵ�ȼ��,����,��ȼ����Сֵ,",
              "#5,����������id,����������,���������(p.u.),������翹(p.u.),��߳�ͷ�ߵ�ѹ(kV),��ͳ�ͷ�ߵ�ѹ(kV),�������ߵ�ѹ(kV),������ڶ�ߵ�ѹ(kV),ƽ��(mH),���������(p.u.),������翹(p.u.),��ͷ����,",
              "#5,Э�����Ʒ�ʽ,����ֱ������(kA),����ֱ������(MW),����ֱ����ѹ(kV),��ȼ��/Ϩ���ǳ�ʼֵ,"],
             [
                 "#6,ֱ����i��ֱ��ĸ�ߺ�,j��ֱ��ĸ�ߺ�,���,����ֱ��ϵͳ��,i���й�(MW),j���й�(MW),����(kA,i-j),����(ŷ),���(��),����(΢��),�����(kA),��·����(0������·;1��������·),ֱ����·����,"],
             ["#7,DC/DC�任��R��ֱ��ĸ�ߺ�,L��ֱ��ĸ�ߺ�,���,R���й�(MW),L���й�(MW),�絼(΢������),����(ŷ),Ԫ������,"]]
    )


def file_format():
    return dict(
        on=online_column(),
        off=offline_column()
    )


def online_column():
    ret = slf.copy()
    ret['l2'] = [i for i in slf['l2'] if i not in ['name_ctrl', 'type_ctrl']]
    ret['l3'] = [i for i in slf['l3'] if i not in ['name_ctrl', 'type_ctrl',
                                                   'uni', 'unj', 'tapmax', 'tapmin',
                                                   'tapun', 'du', 'tappos']]
    ret['l5'] = [i for i in slf['l5'] if i not in ['name_ctrl', 'type_ctrl']]
    ret['l6'] = [i for i in slf['l6'] if i not in ['name_ctrl', 'type_ctrl']]
    ret['lp1'] = [i for i in slf['lp1'] if i not in ['name']]
    ret['lp2'] = [i for i in slf['lp2'] if i not in ['name']]
    ret['lp3'] = [i for i in slf['lp3'] if i not in ['name']]
    ret['lp5'] = [i for i in slf['lp5'] if i not in ['name']]
    ret['lp6'] = [i for i in slf['lp6'] if i not in ['name']]
    ret['s3'] = [i for i in slf['s3'] if i not in ['gm0', 'bm0']]
    ret.update(mlf['dcline'])
    ret.update(mlf['mdc'])
    return ret


def offline_column():
    ret = slf.copy()
    ret['l1'] = [i for i in slf['l1'] if i not in ['island']]
    ret['l6'] = [i for i in slf['l6'] if i not in ['tmp']]
    ret['s3'] = [i for i in slf['s3'] if i not in ['trs_type']]
    ret['ms4'] = [i for i in slf['ms4'] if i not in ['name']]
    ret.update(mlf['dcline'])
    ret.update(mlf['mdc'])
    return ret


def model_column():
    return dict(
        bus=['name', 'vbase', 'area', 'vmax', 'vmin', 'cb1', 'cb3', 'st_name'],
        acline=['no', 'r', 'x', 'b', 'area', 'lc', 'lim', 'name', 'r0', 'x0', 'b0'],
        transformer=['no', 'r', 'x', 'rm', 'xm', 'area', 'tc', 'lim', 'trs_type', 'name',
                     'uni', 'unj', 'tapmax', 'tapmin', 'tapun', 'du',
                     'r0', 'x0', 'tk0', 'gm0', 'bm0'],
        generator=['qmax', 'qmin', 'pmax', 'pmin', 'name',
                   'tg', 'lg', 'tvr', 'lvr', 'tgo', 'lgo', 'tpss', 'lpss',
                   'xdp', 'xdpp', 'x2', 'tj', 'sh', 'ph'],
        load=['no', 'qmax', 'qmin', 'pmax', 'pmin', 'name', 'tp', 'lp', 'k'],
        dcline=list(set(list(chain(*mlf['dcline']['nl4_dcline']))
                        + list(chain(*mlf['dcline']['ns4_dcline'])))),
        dcbus=list(chain(*mlf['mdc']['ml4_dcbus'])) + ['name'],
        vsc=list(set(list(chain(*mlf['mdc']['ml4_vsc'])) + slf['ms4'])),
        lcc=list(chain(*mlf['mdc']['ml4_lcc'])),
        mdcline=list(chain(*mlf['mdc']['ml4_mdcline'])),
        dcsys=[]
    )


def init_column():
    return dict(
        bus=['name'],
        acline=['mark', 'ibus', 'jbus', 'name'],
        transformer=['mark', 'ibus', 'jbus', 'tk', 'name'],
        generator=['mark', 'bus', 'type', 'p0', 'q0', 'v0', 'theta0', 'name'],
        load=['mark', 'bus', 'type', 'p0', 'q0', 'v0', 'theta0', 'name'],
        dcline=['mark', 'ibus', 'jbus', 'name',
                'op', 'qci', 'qcj',
                'pd1', 'vd1', 'a10', 'gama10',
                'pd2', 'vd2', 'a20', 'gama20'],
        dcbus=list(chain(*mlf['mdc']['ml4_dcbus'])),
        vsc=list(chain(*mlf['mdc']['ml4_vsc'])),
        lcc=list(chain(*mlf['mdc']['ml4_lcc'])),
        mdcline=list(chain(*mlf['mdc']['ml4_mdcline']))
    )


def restore_column():
    return dict(
        bus=['name', 'island',
             'bus', 'v', 'theta'],
        acline=['mark', 'ibus', 'jbus', 'name',
                'pi', 'qi', 'pj', 'qj', 'qci', 'qcj',
                'mark0'],
        transformer=['mark', 'ibus', 'jbus', 'tk', 'name', 'tappos',
                     'pi', 'qi', 'pj', 'qj', 'pm', 'qm',
                     'mark0', 'ibus0', 'jbus0'],
        generator=['mark', 'bus', 'type', 'p0', 'q0', 'v0', 'theta0', 'name',
                   'p', 'q'],
        load=['mark', 'bus', 'type', 'p0', 'q0', 'v0', 'theta0', 'name',
              'p', 'q'],
        dcline=list(set(list(chain(*mlf['dcline']['nl4_dcline']))
                        + list(chain(*mlf['dcline']['np4_dcline'])))),
        dcbus=list(set(list(chain(*mlf['mdc']['ml4_dcbus']))
                       + list(chain(*mlf['mdc']['mp4_dcbus'])))),
        vsc=list(set(list(chain(*mlf['mdc']['ml4_vsc']))
                     + list(chain(*mlf['mdc']['mp4_vsc'])))),
        lcc=list(set(list(chain(*mlf['mdc']['ml4_lcc']))
                     + list(chain(*mlf['mdc']['mp4_lcc'])))),
        mdcline=list(set(list(chain(*mlf['mdc']['ml4_mdcline']))
                         + list(chain(*mlf['mdc']['mp4_mdcline'])))),
        dcsys=list(chain(*mlf['mdc']['mp4_dcsys']))
    )


def learning_column():
    return dict(
        bus=['name', 'v'],
        acline=['mark', 'name', 'pi', 'qi'],
        transformer=['mark', 'name', 'trs_type', 'pi', 'qi'],
        generator=['mark', 'name', 'p', 'q', 'v'],
        load=['mark', 'name'],
        dcline=['mark', 'name', 'op', 'pd1i_pu', 'qd1i_pu', 'pd1j_pu', 'qd1j_pu',
                'pd2i_pu', 'qd2i_pu', 'pd2j_pu', 'qd2j_pu'],
        vsc=['mark', 'name', 'p_sys', 'q_sys'],
        lcc=['mark', 'name', 'p_sys', 'q_sys'],
        mdcline=['mark', 'name', 'pi']
    )


def useless_column():
    return dict(
        bus=[],
        acline=['type', 'pl', 'cb', 'cl', 'vqp', 'name_ctrl', 'type_ctrl'],
        transformer=['area', 'type', 'pl', 'tp', 'cb', 'cl', 'vqp', 'theta',
                     'j', 'name_ctrl', 'type_ctrl', 'gm0', 'bm0'],
        generator=['pl', 'cb', 'cl', 'vqp', 'k', 'name_ctrl', 'type_ctrl',
                   'xdp', 'xdpp', 'x2'],
        load=['pl', 'cb', 'cl', 'vqp', 'name_ctrl', 'type_ctrl', 'tmp']
    )


def index_dict():
    return dict(
        l1=['bus'], l2=['no'], l3=['no'], l5=['bus'], l6=['bus', 'no'], nl4=['no'],
        lp1=['bus'], lp2=['no'], lp3=['no'], lp5=['bus'], lp6=['bus', 'no'], np4=['no'],
        s1=['bus'], s2=['no'], s3=['no'], s5=['bus'], s6=['bus', 'no'], ns4=['no'],
        ms4=['no'],
        bus=['bus'], acline=['no'], transformer=['no'],
        generator=['bus'], load=['bus', 'no'],
        dcline=['no'],
        dcbus=['bus'], vsc=['no'], lcc=['no'], mdcline=['no'], dcsys=['id']
    )


def name_index_dict():
    return dict(
        bus=['name'], acline=['name'], transformer=['name', 'trs_type'],
        generator=['name'], load=['name', 'no'], dcline=['name', 'no'],
        dcbus=['name'], vsc=['name'], lcc=['name'], mdcline=['name'], dcsys=['id']
    )


def output_format():
    return dict(
        default={'O': ('', '\'%s\''), 'i': (0, '%d'), 'f': (0., '%.6f')},
        l1={'name': ('', '\'%s\''), 'vbase': (1., '%10.4f'), 'area': (0, '%10d'),
            'vmax': (0., '%10.4f'), 'vmin': (0., '%10.4f'), 'cb1': (0., '%10.4f'),
            'cb3': (0., '%10.4f'), 'st_name': ('', '\'%s\''), 'island': (-1, '%4d')},
        l2={'mark': (0, '%3d'), 'ibus': (None, '%6d'), 'jbus': (None, '%6d'),
            'no': (None, '%6d'), 'r': (0., '%15.6f'), 'x': (0.00001, '%15.6f'),
            'b': (0., '%15.6f'), 'area': (0, '%10d'), 'type': (0, '%2d'),
            'pl': (0, '%2d'), 'cb': (0, '%4d'), 'cl': (0, '%4d'),
            'vqp': (0., '%15.6f'), 'lc': (0., '%15.6f'), 'lim': (100., '%6.2f'),
            'name_ctrl': ('', '\'%s\''), 'type_ctrl': (0, '%2d'),
            'name': ('', '\'%s\'')},
        l3={'mark': (0, '%3d'), 'ibus': (None, '%6d'), 'jbus': (None, '%6d'),
            'no': (None, '%6d'), 'r': (0., '%15.6f'), 'x': (0.00001, '%15.6f'),
            'tk': (1., '%15.6f'), 'rm': (0., '%15.6f'), 'xm': (0., '%15.6f'),
            'area': (0, '%10d'), 'type': (0, '%4d'), 'pl': (0, '%4d'),
            'tp': (0, '%4d'), 'cb': (0, '%4d'), 'cl': (0, '%4d'),
            'vqp': (0., '%15.6f'), 'theta': (0., '%15.6f'),
            'tc': (0., '%15.6f'), 'lim': (100., '%6.2f'),
            'id': (0, '%2d'), 'j': (0, '%2d'), 'trs_type': (1, '%2d'),
            'name_ctrl': ('', '\'%s\''), 'type_ctrl': (0, '%3d'),
            'name': ('', '\'%s\''), 'uni': (0., '%15.6f'), 'unj': (0., '%15.6f'),
            'tapmax': (0, '%3d'), 'tapmin': (0, '%3d'), 'tapun': (0, '%3d'),
            'du': (0., '%15.6f'), 'tappos': (0, '%3d')},
        l5={'mark': (0, '%3d'), 'bus': (None, '%6d'), 'type': (-2, '%4d'),
            'p0': (0., '%15.6f'), 'q0': (0., '%15.6f'),
            'v0': (1.0, '%15.6f'), 'theta0': (0., '%15.6f'),
            'qmax': (0., '%15.6f'), 'qmin': (0., '%15.6f'),
            'pmax': (0., '%15.6f'), 'pmin': (0., '%15.6f'),
            'pl': (0, '%4d'), 'cb': (0, '%4d'), 'cl': (0, '%4d'),
            'vqp': (0., '%15.6f'), 'k': (0, '%3d'), 'name_ctrl': ('', '\'%s\''),
            'type_ctrl': (0, '%3d'), 'name': ('', '\'%s\'')},
        l6={'mark': (0, '%3d'), 'bus': (None, '%6d'), 'no': (None, '%6d'),
            'type': (1, '%3d'), 'p0': (0., '%15.6f'), 'q0': (0., '%15.6f'),
            'v0': (1.0, '%15.6f'), 'theta0': (0, '%15.6f'),
            'qmax': (0., '%15.6f'), 'qmin': (0., '%15.6f'),
            'pmax': (0., '%15.6f'), 'pmin': (0., '%15.6f'),
            'pl': (0, '%4d'), 'cb': (0, '%4d'), 'cl': (0, '%4d'),
            'vqp': (0., '%15.6f'), 'name_ctrl': ('', '\'%s\''),
            'type_ctrl': (0, '%3d'), 'tmp': (0, '%3d'), 'name': ('', '\'%s\'')},
        nl4={'mark': (None, '%3d'), 'ibus': (None, '%6d'), 'jbus': (None, '%6d'),
             'no': (None, '%6d'), 'area': (0, '%6d'), 'name': ('', '\'%s\''),
             'rpi': (0., '%15.6f'), 'lpi': (0., '%15.6f'), 'rpj': (0., '%15.6f'),
             'lpj': (0., '%15.6f'), 'rl': (0., '%15.6f'), 'll': (0., '%15.6f'),
             'rei': (0., '%15.6f'), 'rej': (0., '%15.6f'),
             'lsi': (0., '%15.6f'), 'lsj': (0., '%15.6f'), 'vdn': (0., '%15.6f'),
             'vhi': (0., '%15.6f'), 'vli': (0., '%15.6f'), 'bi': (0, '%4d'),
             'sti': (0., '%15.6f'), 'rti': (0., '%15.6f'), 'xti': (0., '%15.6f'),
             'vtimax': (0., '%15.6f'), 'vtimin': (0., '%15.6f'),
             'ntapi': (0, '%3d'), 'r0i_pu': (0., '%15.6f'), 'x0i_pu': (0., '%15.6f'),
             'vhj': (0., '%15.6f'), 'vlj': (0., '%15.6f'), 'bj': (0, '%4d'),
             'stj': (0., '%15.6f'), 'rtj': (0., '%15.6f'), 'xtj': (0., '%15.6f'),
             'vtjmax': (0., '%15.6f'), 'vtjmin': (0., '%15.6f'),
             'ntapj': (0, '%3d'), 'r0j_pu': (0., '%15.6f'), 'x0j_pu': (0., '%15.6f'),
             'op': (None, '%4d'), 'qci': (0., '%15.6f'), 'qcj': (0., '%15.6f'),
             'pd1': (0., '%15.6f'), 'vd1': (0., '%15.6f'), 'a1min': (5., '%15.6f'),
             'a10': (15., '%15.6f'), 'gama1min': (5., '%15.6f'),
             'gama10': (17., '%15.6f'), 'pd2': (0, '%15.6f'), 'vd2': (0, '%15.6f'),
             'a2min': (5., '%15.6f'), 'a20': (15., '%15.6f'),
             'gama2min': (5., '%15.6f'), 'gama20': (17., '%15.6f')},
        lp1={'bus': (None, '%6d'), 'v': (None, '%15.10f'),
             'theta': (None, '%15.10f'), 'name': ('', '\'%s\'')},
        lp2={'ibus': (None, '%6d'), 'jbus': (None, '%6d'), 'no': (None, '%6d'),
             'pi': (0., '%15.5f'), 'qi': (0., '%15.5f'), 'pj': (0., '%15.5f'),
             'qj': (0., '%15.5f'), 'qci': (0., '%15.5f'), 'qcj': (0., '%15.5f'),
             'name': ('', '\'%s\'')},
        lp3={'ibus': (None, '%6d'), 'jbus': (None, '%6d'), 'no': (None, '%6d'),
             'pi': (0., '%15.5f'), 'qi': (0., '%15.5f'), 'pj': (0., '%15.5f'),
             'qj': (0., '%15.5f'), 'pm': (0., '%15.5f'), 'qm': (0., '%15.5f'),
             'name': ('', '\'%s\'')},
        lp5={'bus': (None, '%6d'), 'p': (None, '%20.10f'), 'q': (None, '%20.10f'),
             'no': (0, '%6d'), 'name': ('', '\'%s\'')},
        lp6={'bus': (None, '%6d'), 'no': (None, '%6d'),
             'p': (None, '%20.10f'), 'q': (None, '%20.10f'), 'name': ('', '\'%s\'')},
        s1={'name': ('', '\'%s\'')},
        np4={'ibus': (None, '%6d'), 'jbus': (None, '%6d'), 'no': (None, '%6d'),
             'area': (None, '%6d'), 'name': ('', '\'%s\''),
             'op': (None, '%6d'), 'id10_pu': (0., '%16.9f'), 'id20_pu': (0., '%16.9f'),
             'vaci_pu': (0., '%16.9f'), 'vacj_pu': (0., '%16.9f'),
             'pd1i_pu': (0., '%16.9f'), 'qd1i_pu': (0., '%16.9f'),
             'vd10i_pu': (0., '%16.9f'), 'tk1i_pu': (0., '%16.9f'),
             'a10r': (0., '%16.9f'), 'ttk1i': (0., '%16.9f'),
             'pd1j_pu': (0., '%16.9f'), 'qd1j_pu': (0., '%16.9f'),
             'vd10j_pu': (0., '%16.9f'), 'tk1j_pu': (0., '%16.9f'),
             'ttk1j': (0., '%16.9f'), 'pd2i_pu': (0., '%16.9f'),
             'qd2i_pu': (0., '%16.9f'), 'vd20i_pu': (0., '%16.9f'),
             'tk2i_pu': (0., '%16.9f'), 'a20r': (0., '%16.9f'),
             'ttk2i': (0., '%16.9f'), 'pd2j_pu': (0., '%16.9f'),
             'qd2j_pu': (0., '%16.9f'), 'vd20j_pu': (0., '%16.9f'),
             'tk2j_pu': (0., '%16.9f'), 'ttk2j': (0., '%16.9f'),
             'vbi': (0., '%16.9f'), 'vbj': (0., '%16.9f'), 'vdb': (0., '%16.9f'),
             'idb': (0., '%16.9f'), 'zdb': (0., '%16.9f'), 'ldb': (0., '%16.9f'),
             'tkbi': (0., '%16.9f'), 'tkbj': (0., '%16.9f'), 'rpi_pu': (0., '%16.9f'),
             'lpi_pu': (0., '%16.9f'), 'rpj_pu': (0., '%16.9f'), 'lpj_pu': (0., '%16.9f'),
             'rl_pu': (0., '%16.9f'), 'll_pu': (0., '%16.9f'), 'rei_pu': (0., '%16.9f'),
             'rej_pu': (0., '%16.9f'), 'lsi_pu': (0., '%16.9f'), 'lsj_pu': (0., '%16.9f'),
             'xci_pu': (0., '%16.9f'), 'xcj_pu': (0., '%16.9f'),
             'tkimax_pu': (0., '%16.9f'), 'tkimin_pu': (0., '%16.9f'),
             'tkjmax_pu': (0., '%16.9f'), 'tkjmin_pu': (0., '%16.9f'),
             'qcip_pu': (0., '%16.9f'), 'qcjp_pu': (0., '%16.9f')},
        s2={'mark0': (0, '%3d'), 'ibus': (None, '%6d'), 'jbus': (None, '%6d'),
            'no': (None, '%6d'), 'r0': (0., '%15.6f'), 'x0': (0., '%15.6f'),
            'b0': (0., '%15.6f'), 'name': ('', '\'%s\'')},
        s3={'mark0': (0, '%3d'), 'ibus0': (None, '%6d'), 'jbus0': (None, '%6d'),
            'no': (None, '%6d'), 'r0': (0., '%15.6f'), 'x0': (0.00001, '%15.6f'),
            'tk0': (1.0, '%15.6f'), 'gm0': (0., '%15.6f'), 'bm0': (0., '%15.6f'),
            'trs_type': (1, '%2d'), 'name': ('', '\'%s\'')},
        s5={'mark': (0, '%3d'), 'bus': (None, '%6d'),
            'tg': (0, '%3d'), 'lg': (0, '%4d'), 'tvr': (0, '%3d'), 'lvr': (0, '%4d'),
            'tgo': (0, '%3d'), 'lgo': (0, '%4d'), 'tpss': (0, '%3d'), 'lpss': (0, '%4d'),
            'xdp': (0.2, '%15.6f'), 'xdpp': (0.1, '%15.6f'), 'x2': (0.1, '%15.6f'),
            'tj': (9999., '%15.6f'), 'sh': (1100., '%15.6f'), 'ph': (1000., '%15.6f'),
            'name': ('', '\'%s\'')},
        s6={'mark': (0, '%3d'), 'bus': (None, '%6d'), 'no': (None, '%6d'),
            'tp': (0, '%4d'), 'lp': (0, '%4d'), 'k': (0, '%3d'), 'name': ('', '\'%s\'')},
        ns4={'mark': (None, '%6d'), 'ibus': (None, '%6d'), 'jbus': (None, '%6d'),
             'no': (None, '%6d'), 'dcm': (None, '%6d'), 'name': ('', '\'%s\''),
             'reg_type': (0, '%6d'), 'reg_fon': (0, '%6d'), 'reg_par': (None, '%6d'),
             'reg_set': (0., '%15.6f'), 'amax': (0., '%15.6f'), 'amin': (0., '%15.6f'),
             'ireg_fon': (0, '%6d'), 'ireg_par': (0, '%6d'), 'ireg_set': (0., '%15.6f'),
             'vreg_fon': (0, '%6d'), 'vreg_par': (0, '%6d'), 'vreg_set': (0., '%15.6f'),
             'greg_fon': (0, '%6d'), 'greg_par': (0, '%6d'), 'greg_set': (0., '%15.6f'),
             'bmax': (0., '%15.6f'), 'bmin': (0., '%15.6f'), 'im': (0., '%15.6f'),
             'v_low': (0., '%15.6f'), 'idmax': (0., '%15.6f'), 'idmin': (0., '%15.6f'),
             'dki': (0., '%15.6f'), 'dkj': (0., '%15.6f'), 'dti': (0., '%15.6f'),
             'dtj': (0., '%15.6f'), 'v_relay': (0., '%15.6f'), 't_relay': (0., '%15.6f'),
             'type_fdc': (0, '%6d'), 'k_fdc': (0, '%6d'), 'ts_fdc': (0., '%15.6f'),
             'te_fdc': (0., '%15.6f'), 'tdo': (0., '%15.6f'), 'tda': (0., '%15.6f'),
             'tdb': (0., '%15.6f'), 'tdc': (0., '%15.6f'), 'n_rs': (0, '%6d'),
             'v_rs': (0., '%15.6f')}
    )


def get_all_column(types, file_columns):
    all_columns = {}
    format_keys = format_key()
    for t in types:
        files = [ex for ex in format_keys[t] if ex is not None]
        columns = []
        for ex in files:
            if isinstance(format_keys[ex], list):
                columns.extend(list(chain(*file_columns[ex + '_' + t])))
            else:
                columns.extend(file_columns[ex])
        all_columns[t] = list(set(columns))
    return all_columns
