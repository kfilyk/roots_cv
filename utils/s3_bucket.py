import boto3
from PIL import Image
from io import BytesIO


class S3Bucket:
    def __init__(self, bucket_name, region_name="ca-central-1"):
        self.s3 = boto3.resource("s3", region_name)
        self.bucket = self.s3.Bucket(bucket_name)
        
    def __len__(self):
        return sum(1 for _ in self.bucket.objects.all())
    
    def __iter__(self):
        for obj in self.bucket.objects.all():
            yield obj  
    
    def __getitem__(self, key):
        obj = self.bucket.Object(key)
        response = obj.get()
        file_stream = response["Body"]
        content_type = response["ContentType"]
        if content_type.startswith("image/"):
            return Image.open(file_stream).convert("RGB")
        if content_type.endswith(("/octet-stream", "/json")):
            return file_stream.read().decode('utf-8')
        return file_stream
    
    def __setitem__(self, key, value):
        obj = self.bucket.Object(key)
        value_type = type(value)
        if value_type == str:
            obj.put(Body=bytes(value.encode('UTF-8')))
        elif value_type == Image.Image:
            file_stream = BytesIO()
            value.save(file_stream, format="JPEG")
            obj.put(Body=file_stream.getvalue(), ContentType="image/jpeg")
        else:
            raise ValueError(f"__setitem__ only accepts a value of type str and PIL.Image.Image. It was given {type(value)}.")
    
    def filter(self, prefix):
        return S3Collection(self.bucket, prefix)
    
    def delete(self, **kwargs):
        if len(kwargs) != 1:
            raise ValueError(f"delete only accepts one keyword argument. It was given {len(kwargs)}.")
        keyword = next(iter(kwargs))
        if keyword == "key":
            self.bucket.Object(kwargs["key"]).delete()
        elif keyword == "prefix":
            self.bucket.objects.filter(Prefix=kwargs["prefix"]).delete()
        else:
            raise ValueError(f'delete only accepts "key" or "prefix" keyword arguments. It was given {keyword}.')


class S3Collection:
    def __init__(self, bucket, prefix):
        self.collection = bucket.objects.filter(Prefix=prefix)
        
    def __len__(self):
        return sum(1 for _ in self.collection)
    
    def __iter__(self):
        for obj in self.collection:
            yield obj  