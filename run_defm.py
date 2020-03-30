if __name__ == "__main__":
    from pc_deform import deform_point_cloud_dislocation as deform_v1
    from disloc import deform_point_cloud_dislocation as deform_v2

    strike = 0
    dip = 89.999
    depth = 0.001
    slip_ss = 0
    slip_ds = 0

    print('Processing model 1...')
    infn = 'data/HSLSurvey101319_utm_thin100.csv'
    outfn = 'output/test1.csv'
    deform_v1(infn, outfn, strike=strike, dip=dip, depth=depth, slip_ss=slip_ss, slip_ds=slip_ds)

    print('Processing model 2...')
    outfn = 'output/test2.csv'
    deform_v2(infn, outfn, strike=strike, dip=dip, depth=depth, slip_ss=slip_ss, slip_ds=slip_ds)

