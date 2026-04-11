from logging import log
from typing import Optional, Union
from pydantic import BaseModel
import requests
MyResponse = Union[requests.Response, BaseModel]

class URLCallerResult:
    def __init__(
            self,
            response: Optional[MyResponse] = None,
            ):
        self.response = response
    
    @property
    def is_empty(self) -> bool:
        """
        Check is the URLCallerResult is empty.

        Returns:
            bool: True if the URLCallerResult is empty, False otherwise.
        """
        return self.response is None
    
    @property
    def json(self):
        """
        JSON content of the response, if any.

        If no JSON content is available, None is returned.
        """
        return None if self.response is None else self.response.json()
    
    @property
    def status(self) -> Optional[int]:
        """
        HTTP status code of the response, if any.

        If no response is available, None is returned.
        """
        return None if self.response is None else self.response.status_code


class URLCaller(BaseModel):
    headers: dict = {}
    verify: bool = True

    def perform_single_call(
            self,
            url: str,
            verb: Optional[str] = "get",
            **extra,
    )-> URLCallerResult:
        """
        Perform a single HTTP call to the specified URL using the given verb.

        Args:
            url (str): 
                The URL to call.
            verb (str):
                The HTTP verb to use (e.g., 'get', 'post', 'put'). Defaults to 'get'.
            extra: 
                Additional arguments to pass to the request function.
        
        Returns:
            An URLCallerResult object containing the response.
        """
        request_args = {
            "url": url,
            "verify": self.verify,
            "headers": self.headers,
            **extra,
        }

        verb = verb.lower()
        if verb == "get":
            fun = requests.get
        elif verb == "post":
            fun = requests.post
        elif verb == "put":
            fun = requests.put
        elif verb == "delete":
            fun = requests.delete
        else:
            raise ValueError(f"Unsupported HTTP verb: {verb}")
        
        try:
            return URLCallerResult(response=fun(**request_args))

        except ConnectionError as e:
            raise ConnectionError(f"Connection error occurred: {e}")