import openvino as ov
core = ov.Core()
model_path = "./models_tmp/deepfilternet3/enc.xml"
model = core.read_model(model_path)

hop_size = 1 # 480
model.reshape({
    "feat_erb": ov.PartialShape((1, 1, hop_size, 32)), 
    "feat_spec": ov.PartialShape((1, 2, hop_size, 96))})
ov.save_model(model, "../models/enc-static.xml", True)



model_path = "./models_tmp/deepfilternet3/erb_dec.xml"
model = core.read_model(model_path)

model.reshape({
    "emb": ov.PartialShape((1, hop_size, 512)),
    "e0": ov.PartialShape((1, 64, hop_size, 32)),
    "e1": ov.PartialShape((1, 64, hop_size, 16)),
    "e2": ov.PartialShape((1, 64, hop_size, 8)),
    "e3": ov.PartialShape((1, 64, hop_size, 8)),
})
ov.save_model(model, "../models/erb_dec-static.xml", True)


model_path = "./models_tmp/deepfilternet3/df_dec.xml"
model = core.read_model(model_path)

model.reshape({
    "emb": ov.PartialShape((1, hop_size, 512)),
    "c0": ov.PartialShape((1, 64, hop_size, 96))})
ov.save_model(model, "../models/df_dec-static.xml", True)

print("OpenVINO models resized and saved in ../models")
