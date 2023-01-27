Basic API interface in front of the pyciemss code

This is really just intended to be a kicking off point and template for further development. Hopefully everything is included that is needed to complete the functionality.


## How to run:

The package can be installed in editing mode using pip:

```bash
cd api
pip install -e .
```
This should install all requirements.

The API runs via uvicorn with the following command:

```uvicorn pyciemss_api.main:api --reload```

(The --reload automatically reloads the api when files change which is very convenient for development)

Test are runnable via `pytest`

## Known issues

* Issues related to serialization: 
  * The serialization of models is currently hex encoded, but switching the encoding to base64 would decrease the payload size when dealing with the models.
  * There is a very naive serialization/deserialization of tensors in which they are just converted to/from lists. This almost certainly loses information and a suitable replacement should be found.
  * Then /infer_parameters and /intervene endpoints do not currently return a useful response as it is unclear how best to serialize the relevant returns.
* The API endpoints are probably at a lower level than what is ideal. But the framework layed out should allow the further addition of other endpoints.
* This was obviously thrown together, so probably quite a bit of clean-up to do within the codebase.