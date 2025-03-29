# Backend APIs

### POST /summarize/v1
Request
```json
{
    "file": "string"
}
```
Response
```json
{
    "data": "string",
    "metaData": {
        "totalTime": "int",
        "transferTime": "int",
        "speechToTextTime": "int",
        "textSummarizationTime": "int" 
    }
}
```


### POST /upload
Request
```json
{
    "file": "file",
    "metadata": {
	"file_type": "string",
	"file_name": "string",
	"user_id": "string"
    }
}
```
Response
```json
{
    "file_path": "string"
}
```