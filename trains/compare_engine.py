import IPython
from ipywidgets import widgets
from trex import *

# Configure a wider output (for the wide graphs)
set_wide_display()

engine_name_0 = "yolov8m_fp16/yolov8m.onnx.engine"
engine_name_1 = "yolov8m_int8/yolov8m.onnx.engine"
engine_name_2 = "qat_old/qat/qat.onnx.engine"
engine_name_3 = "modified/modified.onnx.engine"

plan0 = EnginePlan(f'{engine_name_0}.graph.json', f'{engine_name_0}.profile.json', f"{engine_name_0}.profile.metadata.json")
plan1 = EnginePlan(f'{engine_name_1}.graph.json', f'{engine_name_1}.profile.json', f"{engine_name_1}.profile.metadata.json")
plan2 = EnginePlan(f'{engine_name_2}.graph.json', f'{engine_name_2}.profile.json', f"{engine_name_2}.profile.metadata.json")
plan3 = EnginePlan(f'{engine_name_3}.graph.json', f'{engine_name_3}.profile.json', f"{engine_name_3}.profile.metadata.json")
plans = [plan0, plan1, plan2,plan3]
#
# compare_engines_summaries_tbl(plans, orientation='vertical')
# compare_engines_overview(plans)
#
# compare_engines_layer_latencies(
#     plan1, plan2,
#     # Allow for 3% error grace threshold when color highlighting performance differences
#     threshold=0.03,
#     # Inexact matching uses only the layer's first input and output to match to other layers.
#     exact_matching=True)
# compare_engines_layer_details(plans[0], plans[1])
#
# plan0.summary()
# plan1.summary()
# plan2.summary()

# plot_engine_timings(timing_json_file= f"{engine_name_0}.timing.json")
# plot_engine_timings(timing_json_file= f"{engine_name_1}.timing.json")
# plot_engine_timings(timing_json_file= f"{engine_name_2}.timing.json")


# report_card_reformat_overview(plan2)
# report_card_reformat_overview(plan1)
# report_card_reformat_overview(plan0)

# time_pct_by_type = plan1.df.groupby(["type"]).sum()[["latency.pct_time", "latency.avg_time"]].reset_index()
# display_df(time_pct_by_type)
# plotly_bar2(
#     df=time_pct_by_type,
#     title="% Latency Budget Per Layer Type",
#     values_col="latency.pct_time",
#     names_col="type",
#     orientation='h',
#     color='type',
#     colormap=layer_colormap)


charts = []
layer_precisions = group_count(plan2.df, 'precision')
charts.append((layer_precisions, 'Layer Count By Precision', 'count', 'precision'))
layers_time_pct_by_precision = group_sum_attr(plan2.df, grouping_attr='precision', reduced_attr='latency.pct_time')
print(layers_time_pct_by_precision)
charts.append((layers_time_pct_by_precision, '% Latency Budget By Precision', 'latency.pct_time', 'precision'))
plotly_pie2("Precision Statistics", charts, colormap=precision_colormap)







