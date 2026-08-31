import onnx
import logging
import onnx_graphsurgeon as gs
import numpy as np
class OnnxModifier:
    def __init__(self, onnx_path):
        self.graph = gs.import_onnx(onnx.load(onnx_path))
        self.tensor = self.graph.tensors()
        self.nodes = self.graph.nodes

    def remove_nodes(self, remove_node_list):
        for remove_nd in remove_node_list:
            removed = [node for node in self.nodes if node.name == remove_nd][0]
            self.nodes.remove(removed)

        output_tensor = self.graph.outputs[0]

        self.graph.outputs.remove(output_tensor)

    def add_transpose_nodes(self, input_tensor, perm):
        attrs = {"perm": perm}
        transpose_inputs = [input_tensor]
        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        transpose_outputs = [
            gs.Variable(name="%s_transpose_output" % (input_tensor.name), dtype=input_tensor.dtype, shape=output_shape)]

        transpose_node = gs.Node(op="Transpose", name="%s_transpose" % (input_tensor.name), inputs=transpose_inputs,
                                 outputs=transpose_outputs, attrs=attrs)

        self.nodes.append(transpose_node)

        return transpose_node.outputs[0]

    def add_reducemax_nodes(self, input_tensor, axes=2, keepdims=0):
        attrs = {}
        attrs["axes"] = [axes]
        attrs["keepdims"] = keepdims

        reducemax_inputs = [input_tensor]

        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        reducemax_outputs = [
            gs.Variable(name="%s_reducemax_output" % (input_tensor.name), dtype=input_tensor.dtype, shape=output_shape)]

        reducemax_node = gs.Node(op="ReduceMax", name="%s_reducemax" % (input_tensor.name), inputs=reducemax_inputs,
                                 outputs=reducemax_outputs, attrs=attrs)

        self.nodes.append(reducemax_node)

        return reducemax_node.outputs[0]

    def add_argmax_nodes(self, input_tensor, axis=2, keepdims=0):
        attrs = {}
        attrs["axis"] = axis
        attrs["keepdims"] = keepdims
        argmax_inputs = [input_tensor]

        # output_shape
        if input_tensor.shape is not None:
            output_shape = [*input_tensor.shape[:-1]]
        else:
            output_shape = None

        # argmax_outputs
        argmax_outputs = [
            gs.Variable(name="%s_argmax_output" % (input_tensor.name), dtype=np.int64, shape=output_shape)]

        argmax_node = gs.Node(op="ArgMax", name="%s_argmax" % (input_tensor.name), inputs=argmax_inputs,
                              outputs=argmax_outputs, attrs=attrs)

        self.nodes.append(argmax_node)

        return argmax_node.outputs[0]
    def add_yolov8_output(self, tensor, output_name):
        input_node = tensor.inputs[0]
        if output_name == "bbox":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape=["batch", None, 4])]

        elif output_name == "conf":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape=["batch", None])]

        elif output_name == "class_id":
            input_node.outputs = [
                gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape=["batch", None])]

        return input_node.outputs[0]

    def carve_output(self, output_list):
        self.graph.outputs = output_list

    def save_onnx(self, save_path):
        onnx.save(gs.export_onnx(self.graph), save_path)
    def add_slice_node(self, sig_output_tensor, starts=5, ends=np.iinfo(np.int64).max, axes=4, step=1):
        data_input = sig_output_tensor
        starts_input = gs.Constant(name="%s_%d_%d_starts_Constant" % (data_input.name, starts, axes),
                                   values=np.array([starts]))  # 5 is starts point
        ends_input = gs.Constant(name="%s_%d_%d_ends_Constant" % (data_input.name, starts, axes),
                                 values=np.array([ends]))
        axes_input = gs.Constant(name="%s_%d_%d_axes_Constant" % (data_input.name, starts, axes),
                                 values=np.array([axes]))
        step_input = gs.Constant(name="%s_%d_%d_steps_Constant" % (data_input.name, starts, axes),
                                 values=np.array([step]))

        slice_inputs = [data_input, starts_input, ends_input, axes_input, step_input]
        slice_shape = None

        slice_outputs = [
            gs.Variable(name="%s_%d_%dslice_output_0" % (data_input.name, starts, axes), dtype=data_input.dtype,
                        shape=slice_shape)]

        slice_node = gs.Node(op="Slice", name="%s_%d_%d_slice" % (data_input.name, starts, axes), inputs=slice_inputs,
                             outputs=slice_outputs)

        self.nodes.append(slice_node)

        return slice_node.outputs[0]

    def add_nms_plugin_nodes(self, bbox_output, score_output, keep_topk=200, score_threshold=0.1, iou_threshold=0.5):
        attrs = {}
        attrs["class_agnostic"] = 1
        attrs["background_class"] = -1
        attrs["score_activation"] = 0
        attrs["max_output_boxes"] = keep_topk
        attrs["score_threshold"] = score_threshold
        attrs["iou_threshold"] = iou_threshold
        attrs["box_coding"] = 1
        attrs["plugin_version"] = "1"

        batch_size = self.graph.inputs[0].shape[0]
        input_h = self.graph.inputs[0].shape[2]
        input_w = self.graph.inputs[0].shape[3]

        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
        nmsed_boxes = gs.Variable(name="bbox").to_variable(dtype=np.float32, shape=[batch_size, keep_topk, 4])
        nmsed_scores = gs.Variable(name="conf").to_variable(dtype=np.float32, shape=[batch_size, keep_topk])
        nmsed_classes = gs.Variable(name="class_id").to_variable(dtype=np.int32, shape=[batch_size, keep_topk])

        nms_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

        nms_node = gs.Node(
            op="EfficientNMS_TRT",
            attrs=attrs,
            inputs=[bbox_output, score_output],
            outputs=nms_outputs)

        self.nodes.append(nms_node)

        return nms_node.outputs
class Yolov8Modifier:
    # step 1: remove unused nodes
    # step 2: modify bbox node
    # step 3: modify conf, class_id nodes

    def __init__(self, onnx_load_path, modified_onnx_path):
        self.onnx_load_path = onnx_load_path
        self.onnx_save_path = modified_onnx_path

        self.remove_node_list = ["/model.22/Concat_25"]

        self.modify_tensors_list = []
        self._prepare_modify_tensors()

    def _prepare_modify_tensors(self):
        self.modify_tensors_list.append(("/model.22/Mul_5_output_0", "/model.22/Sigmoid_output_0"))

    def run(self):
        logging.info("# ---- ONNX modification starts ---- #")

        onnx_md = OnnxModifier(self.onnx_load_path)
        graph = onnx_md.graph

        onnx_md.remove_nodes(self.remove_node_list)

        tensor_list = self.modify_tensors_list

        # ---- bbox ---- #
        bbox_out_tenseor = onnx_md.add_transpose_nodes(onnx_md.tensor[tensor_list[0][0]], [0, 2, 1])

        # ---- conf & class-id ---- #
        sigmoid_tensor = onnx_md.tensor[tensor_list[0][1]]
        conf_out_tensor = onnx_md.add_reducemax_nodes(sigmoid_tensor, 1, 0)

        cls_id_out_tesnor = onnx_md.add_argmax_nodes(sigmoid_tensor, 1, 0)

        # ---- make last node ---- #
        bbox_last_output = onnx_md.add_yolov8_output(bbox_out_tenseor, "bbox")
        conf_last_output = onnx_md.add_yolov8_output(conf_out_tensor, "conf")
        cls_id_last_output = onnx_md.add_yolov8_output(cls_id_out_tesnor, "class_id")

        # --- make output ---- #
        onnx_md.carve_output([bbox_last_output, conf_last_output, cls_id_last_output])
        onnx_md.save_onnx(self.onnx_save_path)

if __name__ == '__main__':
    modifier = Yolov8Modifier("../trains/qat.onnx", "qat_modify.onnx")
    modifier.run()