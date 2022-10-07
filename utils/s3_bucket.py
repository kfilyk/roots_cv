import boto3
from PIL import Image
from io import BytesIO


class S3Bucket:
    def __init__(self, bucket_name, region_name="ca-central-1"):
        self.s3 = boto3.resource("s3", region_name)
        self.bucket = self.s3.Bucket(bucket_name)
        
    
    def __iter__(self):
        for obj in self.bucket.objects.all():
            yield obj
            
    
    def __getitem__(self, key):
        obj = self.bucket.Object(key)
        response = obj.get()
        file_stream = response["Body"]
        if response["ContentType"] == "image/jpeg":
            return Image.open(file_stream).convert("RGB")
        return file_stream
    
    
    def __setitem__(self, key, value):
        obj = self.bucket.Object(key)
        try:
            file_stream = BytesIO()
            value.save(file_stream, format="JPEG")
            obj.put(Body=file_stream.getvalue())
        except:
            raise ValueError(f"__setitem__ only accepts a value of type PIL.Image.Image. It was given {type(value)}.")
            
            
    def filter(self, prefix):
        return self.bucket.objects.filter(Prefix=prefix).all()
        
    
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
        