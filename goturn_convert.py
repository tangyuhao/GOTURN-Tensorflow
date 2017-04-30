import caffe
import pickle
model = "tracker.prototxt"
weights = "tracker.caffemodel"
net = caffe.Net(model, weights, caffe.TRAIN)
output_dict = {}
for param_name in net.params.keys():
    weights = net.params[param_name][0].data
    bias = net.params[param_name][1].data
    tmp_data = {"weights":weights, "bias":bias}
    output_dict[param_name] = tmp_data

f = open("./pkl/goturn_weights.pkl","w")

pickle.dump(output_dict, f)
f.close()


