import io

import chardet
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


def get_folder_id(drive_service, folder_name, parent_id=None):
    query = f"name = '{folder_name}'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = (
        drive_service.files()
        .list(q=query, spaces="drive", fields="nextPageToken, files(id, name)")
        .execute()
    )
    items = results.get("files", [])
    if not items:
        print(f"No folder found with the name: {folder_name}")
        return None
    else:
        return items[0]["id"]


def read_sheet_as_df(sheets_service, spreadsheet_id, range_name):
    """
    Read google sheet as a Pandas Dataframe using the ssid and range_name.
    Vars:
    spreadsheet_id:str: last bit of sheet url (sheet identifier)
    range_name:str: name of sheet to be read (optional alphanumeric range indicator)
    """
    # Use the Sheets API to get the spreadsheet data
    result = (
        sheets_service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    values = result.get("values", [])

    if not values:
        print("No data found.")
        return pd.DataFrame()
    else:
        # Handle variable number of columns by aligning data with column names
        max_len = max(len(row) for row in values)
        for row in values:
            while len(row) < max_len:
                row.append(None)

        # Create a pandas DataFrame from the values
        df = pd.DataFrame(values[1:], columns=values[0])
        return df
