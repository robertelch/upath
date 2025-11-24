import requests
class Request:

    def __init__(self):
        self.kwargs = {}

    def make(self):
        return requests.request(**self.kwargs)

    def HEADER(self, **kwargs):
        self.kwargs["headers"] = kwargs
        return self

    def JSON_BODY(self, **kwargs):
        self.kwargs["json"] = kwargs
        return self

    def FORM_BODY(self, **kwargs):
        self.kwargs["data"] = kwargs
        return self

    def URL(self, url: str):
        self.kwargs["url"] = url
        return self

    def GET(self):
        self.kwargs["method"] = "GET"
        return self
    
    def POST(self):
        self.kwargs["method"] = "POST"
        return self
    
req = Request().URL("https://example.com").GET()
print(req.make())
