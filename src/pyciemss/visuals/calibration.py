import pandas as pd

from . import vega


def contour_plot_calibration(datasource: pd.DataFrame) -> vega.VegaSchema:
    """Create a contour plot from the passed datasource.

    datasource --  A dataframe ready for rendering.  Should include:
       - time (int)
       - column_names (str)
       - calibration (bool)
       - y  --- Will be shown as a line
       - y1 --- Upper range of values
       - y0 --- Lower range of values
    """
    for column_name in ['time', 'column_names', 'calibration', 'y', 'y1', 'y0']:
      if column_name not in datasource:
            raise ValueError(
                "Either {} doesn't exist".format( column_name)
            )
    schema = vega.load_schema("calibrate_chart.vg.json")

    data = vega.find_keyed(schema["data"], "name", "table")
    del data["url"]
    data["values"] = datasource.to_dict(orient="records")

    options = sorted(datasource["column_names"].unique().tolist())
    var_filter = vega.find_keyed(schema["signals"], "name", "Variable")
    var_filter["bind"]["options"] = options
    var_filter["value"] = options[0]

    return schema
