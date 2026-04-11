from lib.utils.connectors import URLCaller
from lib.utils.environment import get_conf_for
from lib.utils.token_handler import save_token_to_file, get_token_from_cache
from lib.utils.cache import JSONFileCache

from lib.utils.log import logger
log = logger.get_logger()


class Database():
    def __init__(self, app_name: str):
        self.url_caller = URLCaller()
        self.app_name = app_name
        self.conf = get_conf_for(self.app_name)
        

    
    def get_headers_for_jwt(self):
        api_key = self.conf.get("api_key")
        
        if not api_key:
            log.error("API key not found in config")
            return None
        return {
                "apikey": api_key,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
    
    def get_jwt_auth(self):
        """Get JWT auth token, either from cache or by requesting a new one."""

        token = get_token_from_cache(self.app_name)
        if token:
            return token
        
        base_url = self.conf.get("base_url")
        if not base_url:
            raise ValueError("Base URL not found in config")

        auth_endpoint = self.conf.get("auth_endpoint")
        if not auth_endpoint:
            raise ValueError("Auth endpoint not found in config")

        url = f"{base_url}/{auth_endpoint}"
        email = self.conf.get("email")
        if not email:
            raise ValueError("Email not found in config")
        
        password = self.conf.get("password")
        if not password:
            raise ValueError("Password not found in config")
        
        body = {
            "email": email,
            "password": password
            }

        headers = self.get_headers_for_jwt()
        try:
            result = self.url_caller.perform_single_call(url=url, verb="post", json=body, headers=headers)
        
        except Exception as e:
            log.error(f"Authentication request failed: {e}")
        
        data = result.json
        token = data.get("access_token")
        if not token:
            raise ValueError("Authentication failed: No access token received.")
        
        expires_in = data.get("expires_in")
        save_token_to_file(self.app_name, expires_in, token)
        
        return token
    
    def get_headers(self):
        """Get headers with authorization."""

        jwt_token = self.get_jwt_auth()
        headers = self.get_headers_for_jwt()
        headers["Authorization"] = f"Bearer {jwt_token}"

        return headers
    
    def get_data_from_table(self, table_name: str):
        """
        Fetch data from a specified table in the database.
        
        Args:
            table_name (str): The name of the table to fetch data from.

        Returns:
            dict: The JSON response containing the table data.
        """
        headers = self.get_headers()
        base_url = self.conf.get("base_url")
        if not base_url:
            raise ValueError("Base URL not found in config")
        
        url = f"{base_url}/rest/v1/{table_name}"

        try:
            result = self.url_caller.perform_single_call(url=url, headers=headers, verb="get")

        except Exception as e:
            log.error(f"Data fetch request failed: {e}")
            return None
        
        data = result.json
        
        #for testing purposes
        cache = JSONFileCache(name=f"{self.app_name}_{table_name}_data")
        cache.save(data)

        return data
        
    def insert_data_to_table(self, table_name: str, data: dict):
        """
        Add data to a specified table in the database.
        
        Args:
            table_name (str): The name of the table to add data to.
            data (dict): The data to be added to the table.
                        Example: {
                        "some_column": "someValue",
                        "other_column": "otherValue"
                        }

        Returns:
            dict: The JSON response after adding the data.
        """
        headers = self.get_headers()
        base_url = self.conf.get("base_url")
        if not base_url:
            raise ValueError("Base URL not found in config")
        
        url = f"{base_url}/rest/v1/{table_name}"

        try:
            self.url_caller.perform_single_call(url=url, headers=headers, verb="post", json=data)

        except Exception as e:
            log.error(f"Data addition request failed: {e}")
            return None
        
        log.info(f"Data added to {table_name}")

    def delete_data_from_table(self, table_name: str, id: int):
        """
        Delete data from a specified table in the database based on a condition.
        
        Args:
            table_name (str): The name of the table to delete data from.
            id (int): The ID of the record to delete.

        Returns:
            dict: The JSON response after deleting the data.
        """
        headers = self.get_headers()
        base_url = self.conf.get("base_url")
        if not base_url:
            raise ValueError("Base URL not found in config")
        
        condition = f"id=eq.{id}"
        
        url = f"{base_url}/rest/v1/{table_name}?{condition}"

        try:
            self.url_caller.perform_single_call(url=url, headers=headers, verb="delete")

        except Exception as e:
            log.error(f"Data deletion request failed: {e}")
            return None
        
        log.info(f"Data deleted from {table_name} where {condition}: Success")





if __name__ == "__main__":  #pragma: no cover
    db = Database(app_name="grafana_db")
    a = {"id": "in.(650,651)"}
    db.delete_data_from_table(table_name="app_metrics", id=a)
