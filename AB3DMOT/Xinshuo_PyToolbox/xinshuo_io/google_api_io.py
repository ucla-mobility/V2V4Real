# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions for google gpi, such as google sheet
import os, httplib2
from oauth2client import client, tools
from oauth2client.file import Storage
from googleapiclient import discovery

from xinshuo_miscellaneous import isstring, islistoflist, islist

"""
BEFORE RUNNING:
---------------
1. If not already done, enable the Google Sheets API
   and check the quota for your project at
   https://console.developers.google.com/apis/api/sheets
2. Install the Python client library for Google APIs by running
   `pip install --upgrade `
"""

def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    try:
        import argparse
        flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
    except ImportError:
        flags = None

    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def get_sheet_service():
    # If modifying these scopes, delete your previously saved credentials
    # at ~/.credentials/sheets.googleapis.com-python-quickstart.json
    SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
    # Authorize using one of the following scopes:
    #     'https://www.googleapis.com/auth/drive'
    #     'https://www.googleapis.com/auth/drive.file'
    #     'https://www.googleapis.com/auth/drive.readonly'
    #     'https://www.googleapis.com/auth/spreadsheets'
    #     'https://www.googleapis.com/auth/spreadsheets.readonly'    
    #     'https://www.googleapis.com/auth/drive'
    #     'https://www.googleapis.com/auth/drive.file'
    #     'https://www.googleapis.com/auth/spreadsheets'

    CLIENT_SECRET_FILE = 'client_secret.json'
    APPLICATION_NAME = 'Google Sheets API Python'

    # TODO: Change placeholder below to generate authentication credentials. See
    # https://developers.google.com/sheets/quickstart/python#step_3_set_up_the_sample

    # credentials = None
    credentials = get_credentials()
    service = discovery.build('sheets', 'v4', credentials=credentials)

    return service


def update_patchs2sheet(service, sheet_id, starting_position, data, debug=True):
    '''
    update a list of list data to a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        starting_position:      a string existing in the sheet to represent the let-top corner of patch to fill in
        data:                   a list of list data to fill
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        isstring(starting_position), 'the starting position is not correct'
        islistoflist(data), 'the input data is not a list of list'

    # How the input data should be interpreted.
    value_input_option = 'RAW'  # TODO: Update placeholder value.

    value_range_body = {'values': data}
    request = service.spreadsheets().values().update(spreadsheetId=sheet_id, range=starting_position, valueInputOption=value_input_option, body=value_range_body)
    response = request.execute()

def update_row2sheet(service, sheet_id, row_starting_position, data, debug=True):
    '''
    update a list of data to a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        starting_position:      a string existing in the sheet to represent the let-top corner of patch to fill in
        data:                   a of list data to fill
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        isstring(row_starting_position), 'the starting position is not correct'
        islist(data), 'the input data is not a list'

    # How the input data should be interpreted.
    value_input_option = 'RAW'  # TODO: Update placeholder value.

    value_range_body = {'values': [data]}
    request = service.spreadsheets().values().update(spreadsheetId=sheet_id, range=row_starting_position, valueInputOption=value_input_option, body=value_range_body)
    response = request.execute()

def get_data_from_sheet(service, sheet_id, search_range, debug=True):
    '''
    get a list of data from a google sheet continuously

    parameters:
        service:    a service request to google sheet
        sheet_di:   a string to identify the sheet uniquely
        search_range:      a list of position queried 
    '''

    if debug:
        isstring(sheet_id), 'the sheet id is not a string'
        islist(search_range), 'the search range is not a list'

    # print(search_range)
    # How the input data should be interpreted.
    # value_input_option = 'RAW'  # TODO: Update placeholder value.

    # value_range_body = {'values': [data]}
    request = service.spreadsheets().values().batchGet(spreadsheetId=sheet_id, ranges=search_range)

    while True:
        try:
            response = request.execute()
            break
        except:
            continue

    data = list()
    # print(response['valueRanges'])
    for raw_data in response['valueRanges']:
        if 'values' in raw_data:
            data.append(raw_data['values'][0][0])
        else:
            data.append('')

    return data