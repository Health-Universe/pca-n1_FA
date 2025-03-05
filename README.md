# tool-template
template for making navigator tools

- 1 endpoint with a form based post

- caveats:
  - multiselect fields must be specified like this:
```python
from pydantic import Field
from typing import List

multiselect_field: List[str] = Field(
  title="Multiselect Field",
    description="This is a multiselect field",
    json_schema_extra={
        "schema": {
            "items": ["Item 1", "Item 2", "Item 3"],
        }
    }
)
```
  - File fields are not yet supported but will be soon


