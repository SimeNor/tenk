import boto3
import os


def download_and_extract_images():
    BUCKET_NAME = 'celebfaces' # replace with your bucket name

    # enter authentication credentials
    s3 = boto3.resource('s3',
                        aws_access_key_id='',
                        aws_secret_access_key='')

    s3.Bucket(BUCKET_NAME).download_file('imdbFaces.zip', 'imdbFaces.zip')
    s3.Bucket(BUCKET_NAME).download_file('imdb.csv', 'imdb.csv')

    os.system('unzip imdbFaces.zip && rm imdbFaces.zip')
