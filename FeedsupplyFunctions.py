'''
General feed supply equations used throughout AFO.
'''


import numpy as np
import math

import Functions as fun
import PropertyInputs as pinp
import UniversalInputs as uinp


#######################################
#function for feed budget & livestock #
#######################################
def dmd_to_md(dmd):
    '''
    Calculation of megajoules energy per kg dry matter (MD) from dry matter digestibility (DMD).

    Where:

        * MD is megajoules of energy per unit of dry matter,
        * DMD is dry matter digestibility.

    :param dmd: dmd can be a numpy array or a scalar it can be either a percentage or a decimal.
    :returns: M/D in MJ of ME per kg of DM

    Note: this could be expanded to include forage (0.172 * dmd - 1.7)
          and supplement (.133 * dmd + 23.4 ee + 1.32)
          using an extra 'type' input that is default 'herbage'
    '''
    try:
        if (dmd > 1).all() : dmd /= 100 # if dmd is a list or an array and is not a decimal then convert to decimal (in excel 80% is 0.8 in python)
    except:
        if dmd > 1:          dmd /= 100 # if dmd is a scalar and is not a decimal then convert to decimal   ^ alternative would be to convert scalar values to a list (if dmd isinstance not list: dmd=[dmd]) or perhaps type is float]
    return np.maximum(0, 17 * dmd - 2)                # formula 1.13C from SCA 1990 pg 9

def md_to_dmd(md):
    '''
    Calculation of dry matter digestibility (dmd) from megajoule per kg of dry matter (MD).

    Where:

        * DMD is dry matter digestibility.
        * MD is megajoules of energy per unit of dry matter

    :param md: MJ of ME per kg of DM
    :returns: dmd as a decimal
    '''
    return (md+2)/17


def f_effective_mei(dmi, md, threshold, ri=1, eff_above=0.5):
    """Calculate MEI and scale for reduced efficiency if feed quality is above the animal requirements and the
       voluntary feed intake would be greater than that required to meet the target LW profile.
       This would then necessitate rationing the animal or switching to a lower quality feed source for a period
       while the animal lost weight.
       Rationing would not reduce the efficiency of utilising the high quality feed source. AN example of this is
       offering limited quantities of supplementary feed, or using strip grazing to ration intake on green pasture.
       Whereas allowing weight gain and then following with a period of LW loss reduces efficiency because the
       efficiency of LW gain (kg) is lower than the efficiency of maintenance (km).
       The effective mei varies for each feed pool (v) because the feed quality required to meet the target profile
       varies between pools. The target profile is based on the mid-point of the feed pool.

    :param dmi: value or array - Dry matter intake of the feed decision variable (kg).
    :param md: value or array - M/D of the feed (MJ of ME / kg of DM).
    :param threshold: value or array - Diet quality (ME/Vol) required by animals. Below the threshold: effective m/d == m/d
    :param ri: value or array, optional (1.0) - Relative intake of the feed (quality and quantity).
    :param eff_above: value or array, optional (0.5) - Efficiency that energy is used if above required quality, and
                      animals are gaining then losing weight.

    If inputs are provided in arrays then they must be broadcastable.

    :return: ME available to the animal to meet their ME requirements for the target profile, from the quantity of DM consumed in the feed decision variable.

    """
    fev = md * ri
    fev_effective  = np.minimum(fev, threshold + (fev - threshold) * eff_above)
    md_effective = fev_effective / ri
    mei_effective = dmi * md_effective
    return mei_effective


def f_foo_convert(cu3, cu4, foo, pasture_stage, legume=0, cr=None, z_pos=-1, treat_z=False):
    '''
    Adjust FOO for measurement method.

    Depending on the region FOO can be measured differently. For example, in WA when measuring FOO
    feed is cut at a very low level using a scalpel verses NSW where it is cut at a higher level
    with shears. This results in the same amount of feed being valued at a higher FOO in WA.
    The FOO is adjusted to the method which is used in the livestock production equations which
    is the shears method.

    FOO only needs to be adjusted for the intake equations. It does not need to be adjusted for the pasture
    activities because the pasture growth rate will be the same under both FOO measurement methods.
    The base FOO level (level which grazing can not occur below) will account for differences in a
    regions FOO measurement method. Thus in WA the base FOO level is higher than NSW.

    :param cu3: this parameter should already be slice on the c4 axis.
    :param cu4: this parameter should already be slice on the c4 axis.
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[1,…] using c2[0] (finewool merino)
        cr12 = uinp.parameters['i_cr_c2'][12,0]
    else:
        cr12=cr[12, ...]
    ##pasture conversion scenario (convert the region and pasture stage to an index
    ### because the second axis of cu3 is a combination of region & stage)
    conversion_scenario = pinp.sheep['i_region'] * uinp.pastparameters['i_n_pasture_stage'] + pasture_stage
    ##select cu3&4 params for the specified region and stage. Remaining axes are season and formula coefficient (intercept & slope)
    cu3=cu3[..., conversion_scenario]
    cu4=cu4[..., conversion_scenario]
    ##Convert FOO to hand shears measurement
    foo_shears = np.maximum(0, np.minimum(foo, cu3[2] + cu3[0] * foo + cu3[1] * legume))
    ##Estimate height of pasture
    height = np.maximum(0, np.exp(cu4[3] + cu4[0] * foo + cu4[1] * legume + cu4[2] * foo * legume) + cu4[5] + cu4[4] * foo)
    ##Height density (height per unit FOO)
    hd = fun.f_divide(height, foo_shears) #handles div0 (eg if in feedlot with no pasture or adjusted foo is less than 0)
    ##height ratio
    hr = pinp.sheep['i_hr_scalar'] * hd / uinp.pastparameters['i_hd_std']
    ##calc hf
    hf = 1 + cr12 * (hr -1)
    ##apply z treatment
    if treat_z:
        foo_shears = pinp.f_seasonal_inp(foo_shears,numpy=True,axis=z_pos)
        hf = pinp.f_seasonal_inp(hf,numpy=True,axis=z_pos)
    return foo_shears, hf

def f_ra_cs(foo, hf, cr=None, zf=1):
    '''
    CSIRO relative availability of a feed.

    Relative availability is the availability of a feed to livestock. It is the product of two
    components; relative rate of eating (RR) and relative time spent grazing (RT).

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr4 = uinp.parameters['i_cr_c2'][4, 0]
        cr5 = uinp.parameters['i_cr_c2'][5, 0]
        cr6 = uinp.parameters['i_cr_c2'][6, 0]
        cr13 = uinp.parameters['i_cr_c2'][13, 0]
    else:
        cr4 = cr[4, ...]
        cr5 = cr[5, ...]
        cr6 = cr[6, ...]
        cr13 = cr[13, ...]
    ##Relative rate of eating (rr) & Relative time spent grazing (rt)
    try:
        ###Scalar version
        rr = 1 - math.exp(-(1 + cr13 * 1) * cr4 * hf * zf * foo) #*1 is a reminder that this formula could be improved in a future version
        rt = 1 + cr5 * math.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    except:
        ###Numpy version
        rr = 1 - np.exp(-(1 + cr13 * 1) * cr4 * hf * zf * foo) #*1 is a reminder that this formula could be improved in a future version
        rt = 1 + cr5 * np.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    ##Relative availability
    ra = rr * rt
    return ra


def f_rq_cs(dmd, legume, cr=None, sf=0):
    '''
    CSIRO relative ingestibility (rq) of a feed source.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    NOTE 2: sf is currently not used. Will be required if using tropical species
    '''
    ##To work for DMD as a % or a proportion
    try:
        if (dmd >= 1).any() : dmd /= 100
    except:
        if dmd >= 1:          dmd /= 100
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr1 = uinp.parameters['i_cr_c2'][1, 0]
        cr3 = uinp.parameters['i_cr_c2'][3, 0]
    else:
        cr1 = cr[1, ...]
        cr3 = cr[3, ...]
    ##Relative ingestibility
    try:
        ###Scalar version of formula
        rq = max(0.01, min(1, 1 - cr3 * (cr1 - (dmd +  sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    except:
        ###Numpy version of formula
        rq = np.maximum(0.01, np.minimum(1, 1 - cr3 * (cr1 - (dmd +  sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    return rq


def f_ra_mu(foo, hf, zf=1, cu0=None):
    '''
    Murdoch relative availability of a feed.

    Relative availability is the availability of a feed to livestock.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    '''
    ##create scalar cr if not passed in
    if cu0 is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cu0 = uinp.parameters['i_cu0_c2'][0, 0]
    else:
        cu0 = cu0[0, ...]
    ##Relative availability
    try:
        ###Scalar version
        ra = 1 - cu0 ** (hf * zf * foo)
    except:
        ###Numpy version (exact same so not required atm)
        ra = 1 - cu0 ** (hf * zf * foo)
    return ra


def f_rel_intake(ra, rq, legume, cr=None):
    '''
    Relative intake – The proportion of livestock potential intake required to consume 1kg of
    feed given the availability and digestibility.

    Calculation of relative intake includes the effect of feed availability, feed quality and the interaction.
    This function is not called for feeds (such as supplements) that do not have an 'availability' characteristic.
    The calculated RI can be greater than 1 - which implies that actual intake can be greater than potential intake
    This can occur if rq is greater than 1, due to the 'legume' effect on the intercept or if DMD is greater than cr1.

    NOTE: Only pass cr parameter if called from Stock_generator that have a g axis
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[…] using c2[0] (finewool merino)
        cr2 = uinp.parameters['i_cr_c2'][2, 0]
    else:
        cr2=cr[2, ...]

    ##Relative intake

    try:
        ###Scalar version of formula
        ri = max(0.05, ra * rq * (1 + cr2 * ra**2 * legume))
    except:
        ###Numpy version of formula
        ri = np.maximum(0.05, ra * rq * (1 + cr2 * ra**2 * legume))
    return ri
