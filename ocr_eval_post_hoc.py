import numpy as np
import pandas as pd
from pathlib import Path
import fastwer


def _get_ios_eval(to_test_l, gr_l, df, type):
    #col_l = ['tot', 'app_1', 'app_1_num', 'app_2', 'app_2_num', 'app_3', 'app_3_num', 'all', 'cer']
    col_l = ['tot_accu_rate', 'tot_cer',
             'app_1_accu_rate','app_1_cer', 'app_1_num_accu_rate','app_1_num_cer',
             'app_2_accu_rate','app_2_cer', 'app_2_num_accu_rate', 'app_2_num_cer',
             'app_3_accu_rate','app_3_cer', 'app_3_num_accu_rate', 'app_3_num_cer', 'all']

    accu_l = []
    for i in range(len(gr_l)):
        accu_l.append(df.loc[df[to_test_l[i]]==df[gr_l[i]],:].shape[0] / df.shape[0])
        accu_l.append(np.mean(df[[to_test_l[i], gr_l[i]]].apply(lambda x: fastwer.score_sent(str(x[to_test_l[i]]), str(x[gr_l[i]]),
                                                                               char_level=True),
                                          axis=1)) / 100)

    df['sc_ios'] = df[to_test_l].apply(lambda x: x.to_list(), axis=1)
    df['sc_ios_gr'] = df[gr_l].apply(lambda x: x.to_list(), axis=1)
    accu_l.append(df.loc[df['sc_ios']==df['sc_ios_gr'], :].shape[0] / df.shape[0])

    res_tmp = pd.DataFrame(dict(zip(col_l, accu_l)), index=[0])
    res_tmp['type'] = type
    res_tmp['n'] = df.shape[0]
    return res_tmp

def _handle_app_name(x, correct_appname_dict):
    # this is the 1st way: split the appname by space and get the longest part as the new name
    # this way could correct "e tiktok" to 'tiktok' but fails when the app ame is 'castle crush'.
    #if len(x.split(' ')) == 0:
    #    x_out = x
    #else:
        #length_l = [len(i) for i in x.split(' ')]
        #idx = length_l.index(np.max(length_l))
        #x_out = x.split(' ')[idx]

    # use the new correct_appname_dict
    if x in correct_appname_dict.keys():
        x_out = correct_appname_dict[x]
    else:
        x_out = x

    return x_out

def get_ocr_evaluation_result(res,
                              ocr_result_df,
                              android_img_user_link_df,
                              ios_img_user_link_df,
                              handle_app_name,
                              correct_appname_dict):

    '''
    This function input the OCR results and the ground truth and output the accuracy rate of the OCR results for
    the six categories: ios_screentime, ios_activation, ios_notification, android_screentime, android_activation,
    and android_notification.

    res {pandas df}: an empty pd df to save the accuracy output
    ocr_result_df {pandas df}: the OCR results of both the IOS and Android screenshots
    android_img_user_link_df {pandas df}: the crosswalk of image index and url for android screenshots
    ios_img_user_link_df {pandas df}: the crosswalk of image index and url for ios screenshots

    handle_app_name {boolean}: if True, then correct the appname based on the values in correct_appname_dict;
                               if False, do not correct the OCR results
    correct_appname_dict {dict}: The dict of the OCR identified appnames (keys) and the corrected appnames (values).
    '''

    path_oe = Path("D:\\my research\\happyb\\data\\img_evaluation")
    path_happyb = Path('D:\\my research\\happyb\\data')

    ## for ios screentime
    ## read in the groupdtruth
    type_l = ['ios_screentime', 'an_screentime']
    df = pd.read_csv(path_oe / f'{type_l[0]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='ios', :].reset_index()
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total screentime (min)', '# 1 app with the most screentime', 'st_app_1_min',
             '# 2 app with the most screentime', 'st_app_2_min', '# 3 app with the most screentime', 'st_app_3_min', ]]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df['index'] = df['index'].astype('str')

    df = pd.merge(df, gr[['index', 'screenshot']], on=['index'], how='left')
    df_l = ios_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[2_IMG] Question 2 of Survey 14718']]
    df_l.columns = ['name', 'time', 'screenshot']
    df = pd.merge(df, df_l, on='screenshot', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_usage_time_minutes_yesterday',
                '1_most_used_app_name_yesterday', '1_most_used_app_usage_time_minutes_yesterday',
                '2_most_used_app_name_yesterday', '2_most_used_app_usage_time_minutes_yesterday',
                '3_most_used_app_name_yesterday', '3_most_used_app_usage_time_minutes_yesterday',]
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [1,3,5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')

    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_screentime_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_screentime')
    res = pd.concat([res, res_tmp], axis=0)

    ## for an screentime
    ## read in the groupdtruth
    type_l = ['ios_screentime', 'an_screentime']
    df = pd.read_csv(path_oe / f'{type_l[1]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='andriod', :].reset_index()
    gr['index'] = gr['index'].apply(lambda x: x-44)
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total screentime (min)', '# 1 app with the most screentime', 'st_app_1_min',
             '# 2 app with the most screentime', 'st_app_2_min', '# 3 app with the most screentime', 'st_app_3_min', ]]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df = df.iloc[0:22]
    df['index'] = df['index'].astype('int').astype('str')


    df = pd.merge(df, gr[['index', 'screenshot']], on=['index'], how='left')
    df_l = android_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[2_IMG] Question 2 of Survey 14719']]
    df_l.columns = ['name', 'time', 'screenshot']
    df = pd.merge(df, df_l, on='screenshot', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_usage_time_minutes_yesterday',
                '1_most_used_app_name_yesterday', '1_most_used_app_usage_time_minutes_yesterday',
                '2_most_used_app_name_yesterday', '2_most_used_app_usage_time_minutes_yesterday',
                '3_most_used_app_name_yesterday', '3_most_used_app_usage_time_minutes_yesterday',]
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [1,3,5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')

    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_screentime_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_screentime')
    res = pd.concat([res, res_tmp], axis=0)

    ## ios activation
    ## read in the groupdtruth
    type_l = ['ios_activation', 'an_activation']
    df = pd.read_csv(path_oe / f'{type_l[0]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='ios', :].reset_index()
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total number of activation',
           '1st app with the most activations',
           '# number of activations with the 1st app',
           '# 2 app with the most activations',
           '# number of activations with the 2nd app',
           '# 3 app with the most activations',
           '# number of activations with the 3rd app']]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df['index'] = df['index'].astype('str')

    df = pd.merge(df, gr[['index', 'activation']], on=['index'], how='left')
    df_l = ios_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[3_IMG] Question 3 of Survey 14718']]
    df_l.columns = ['name', 'time', 'activation']
    df = pd.merge(df, df_l, on='activation', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_screen_activations_yesterday',
                '1_most_opened_app_name_yesterday', '1_most_opened_app_num_times_opened_yesterday',
                '2_most_opened_app_name_yesterday', '2_most_opened_app_num_times_opened_yesterday',
                '3_most_opened_app_name_yesterday', '3_most_opened_app_num_times_opened_yesterday',]
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [0, 2, 4, 6]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    for i in [1,3,5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_activation_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_activation')

    res = pd.concat([res, res_tmp], axis=0)
    res = res.reset_index(drop=True)


    ## an activation
    ## read in the groupdtruth
    type_l = ['ios_activation', 'an_activation']
    df = pd.read_csv(path_oe / f'{type_l[1]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='andriod', :].reset_index()
    gr['index'] = gr['index'].apply(lambda x: x-44)
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total number of activation',
           '1st app with the most activations',
           '# number of activations with the 1st app',
           '# 2 app with the most activations',
           '# number of activations with the 2nd app',
           '# 3 app with the most activations',
           '# number of activations with the 3rd app']]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df = df.iloc[0:22]
    df['index'] = df['index'].astype('int').astype('str')

    df = pd.merge(df, gr[['index', 'activation']], on=['index'], how='left')
    df_l = android_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[3_IMG] Question 3 of Survey 14719']]
    df_l.columns = ['name', 'time', 'activation']
    df = pd.merge(df, df_l, on='activation', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_screen_activations_yesterday',
                '1_most_opened_app_name_yesterday', '1_most_opened_app_num_times_opened_yesterday',
                '2_most_opened_app_name_yesterday', '2_most_opened_app_num_times_opened_yesterday',
                '3_most_opened_app_name_yesterday', '3_most_opened_app_num_times_opened_yesterday',]
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [0, 2, 4, 6]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('float').astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    for i in [1, 3, 5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_activation_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_activation')

    res = pd.concat([res, res_tmp], axis=0)
    res = res.reset_index(drop=True)

    ## ios notification
    ## read in the groupdtruth
    type_l = ['ios_notification', 'an_notification']
    df = pd.read_csv(path_oe / f'{type_l[0]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='ios', :].reset_index()
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total number of notification',
           '1st app with the most notification',
           '# number of notification with the 1st app',
           '# 2 app with the most notification',
           '# number of notification with the 2nd app',
           '# 3 app with the most notification',
           '# number of notification with the 3rd app']]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df['index'] = df['index'].astype('str')

    df = pd.merge(df, gr[['index', 'notification']], on=['index'], how='left')
    df_l = ios_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[4_IMG] Question 4 of Survey 14718']]
    df_l.columns = ['name', 'time', 'notification']
    df = pd.merge(df, df_l, on='notification', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_notifications_yesterday',
           '1_app_with_most_notifications_yesterday',
           '1_app_notification_count_yesterday',
           '2_app_with_most_notifications_yesterday',
           '2_app_notification_count_yesterday',
           '3_app_with_most_notifications_yesterday',
           '3_app_notification_count_yesterday']
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [0]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].apply(lambda x: str(int(x) if str(x)!='nan' else str(x)))
    for i in [2, 4, 6]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    for i in [1,3,5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')
    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_notification_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='ios_notification')
    res = pd.concat([res, res_tmp], axis=0)
    res = res.reset_index(drop=True)

    # check
    #df[['total_notifications_yesterday', 'tot_sc_min']]
    #df[['1_app_notification_count_yesterday', 'st_app_1_min']]

    ## an notification
    ## read in the groupdtruth
    type_l = ['ios_notification', 'an_notification']
    df = pd.read_csv(path_oe / f'{type_l[1]}.csv')
    gr= pd.read_csv(path_oe / 'happyb_ocr_evaluation_groundtruth_sample.csv')

    gr = gr.loc[gr['phone_type']=='andriod', :].reset_index()
    gr['index'] = gr['index'].apply(lambda x: x-44)
    gr['index'] = gr['index'].astype('str')
    df = df[['name of plot', 'Total number of notification',
           '1st app with the most notification',
           '# number of notification with the 1st app',
           '# 2 app with the most notification',
           '# number of notification with the 2nd app',
           '# 3 app with the most notification',
           '# number of notification with the 3rd app']]
    df.columns = ['index', 'tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    df = df.iloc[0:22]
    df['index'] = df['index'].astype('int').astype('str')

    df = pd.merge(df, gr[['index', 'notification']], on=['index'], how='left')
    df_l = android_img_user_link_df[['Participant ID', 'Session Scheduled Time',
           '[4_IMG] Question 4 of Survey 14719']]
    df_l.columns = ['name', 'time', 'notification']
    df = pd.merge(df, df_l, on='notification', how='left')

    df['time'] = df['time'].apply(lambda x: x[0:10])
    ocr_result_df['time'] = ocr_result_df['time'].apply(lambda x: x[0:10])

    df = pd.merge(df, ocr_result_df, on=['name', 'time'], how='left')
    str_l = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
             '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
             '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday', '3_app_with_most_notifications_yesterday']
    for i in str_l:
        df[i] = df[i].apply(lambda x: str(x).lower())
        if handle_app_name:
            df[i] = df[i].apply(lambda x: _handle_app_name(x, correct_appname_dict))

    to_test_l =['total_notifications_yesterday',
           '1_app_with_most_notifications_yesterday',
           '1_app_notification_count_yesterday',
           '2_app_with_most_notifications_yesterday',
           '2_app_notification_count_yesterday',
           '3_app_with_most_notifications_yesterday',
           '3_app_notification_count_yesterday']
    gr_l = ['tot_sc_min', 'st_app_1', 'st_app_1_min','st_app_2', 'st_app_2_min','st_app_3', 'st_app_3_min']
    for i in [0, 2, 4, 6]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        #df[gr_l[i]] = df[gr_l[i]].apply(lambda x: str(int(x) if str(x)!='nan' else str(x)))
        df[gr_l[i]] = df[gr_l[i]].apply(lambda x: str(x))
    for i in [1,3,5]:
        df[to_test_l[i]] = df[to_test_l[i]].astype('str')
        df[gr_l[i]] = df[gr_l[i]].astype('str')

    if handle_app_name:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_notification_name_clean')
    else:
        res_tmp = _get_ios_eval(to_test_l, gr_l, df, type='an_notification')

    res = pd.concat([res, res_tmp], axis=0)
    res = res.reset_index(drop=True)

    return res


if __name__=='__main__':
    path_oe = Path("D:\\my research\\happyb\\data\\img_evaluation")
    path_happyb = Path('D:\\my research\\happyb\\data')

    ## read in the img-user time link
    file_to_process = path_happyb / 'activity_response_2207_14719_3.csv'
    file_to_process2 = path_happyb / 'activity_response_2207_14718_2.csv'
    android_img_user_link_df = pd.read_csv(file_to_process) # andriod
    ios_img_user_link_df = pd.read_csv(file_to_process2) # ios

    ## read in the OCR results
    ios_ocr_result = pd.read_csv(path_happyb / 'resulting_survey_responses_14718_IOS_final.csv')
    android_ocr_result = pd.read_csv(path_happyb / 'results_survey_responses_14719_android.csv')
    ios_ocr_result['ios_or_an'] = 'ios'
    android_ocr_result['ios_or_an'] = 'an'
    ocr_result_df = pd.concat([ios_ocr_result, android_ocr_result], axis=0)
    ocr_result_df = ocr_result_df.rename(columns={'Name': 'name', 'Scheduled Time': 'time'})

    '''## output a list of all appnames and the frequence of the appnames identified by OCR for post-hoc
    appname_fields = ['1_most_used_app_name_yesterday', '2_most_used_app_name_yesterday', '3_most_used_app_name_yesterday',
                      '1_most_opened_app_name_yesterday', '2_most_opened_app_name_yesterday', '3_most_opened_app_name_yesterday',
                      '1_app_with_most_notifications_yesterday', '2_app_with_most_notifications_yesterday',
                      '3_app_with_most_notifications_yesterday']
    ph = pd.DataFrame()
    for i in appname_fields:
        ocr_result_df[i] = ocr_result_df[i].apply(lambda x: str(x).lower())
        ph_temp = ocr_result_df[[i,'name']].groupby(i).count().reset_index().sort_values(by=['name'], ascending=False)
        ph_temp.columns = ['appname', 'freq']
        ph = pd.concat([ph, ph_temp], axis=0)
    
    ph = ph.groupby(['appname']).agg({'freq':'sum'}).reset_index().sort_values(by=['freq'], ascending=False)
    ph.to_csv(path_happyb / 'post_hoc_screentime_appname.csv', index=False)
    ## manually correct the appnames then import again'''

    ## read in the df with the corrected appnames
    ph = pd.read_csv(path_happyb / 'post_hoc_screentime_appname.csv')
    ph['correct_appname'] = ph[['appname',
                                'correct_appname']].apply(lambda x: x['appname'] if str(x['correct_appname'])=='nan' else x['correct_appname'], axis=1)
    correct_appname_dict = dict(zip(ph['appname'], ph['correct_appname']))

    ## run the function to get the ocr accuracy without or with the correction in the appnames
    ## not handle_app_name: the raw accuracy
    handle_app_name = False
    res = pd.DataFrame()
    res = get_ocr_evaluation_result(res, ocr_result_df,
                                    android_img_user_link_df, ios_img_user_link_df,
                                    handle_app_name, correct_appname_dict)
    res.to_csv(path_oe / 'ocr_eval_res_not_handle_app_name.csv', index=False)

    ## handle app name: the improved accuracy
    handle_app_name = True
    res = pd.DataFrame()
    res = get_ocr_evaluation_result(res, ocr_result_df,
                                    android_img_user_link_df, ios_img_user_link_df,
                                    handle_app_name, correct_appname_dict)
    res.to_csv(path_oe / 'ocr_eval_res_handle_app_name.csv', index=False)


