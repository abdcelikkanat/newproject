from src.utils import *


# contig_names, contig_depth_means, contig_depth_vars = compute_coverage(bam_file_path="dataset/1-NP.bam")
#
# print(contig_names[0])
# print(contig_depth_means[0])
# print(contig_depth_vars[0])


from src.pairdataset import PairDataset
from src.tokenizer import KmerTokenizer

# k=4
# tokenizer = KmerTokenizer(k=k, counting_method = "canonical")
# dataset = PairDataset(
#     file_path="dataset/example.fasta",
#     tokenizer=tokenizer, neg_sample_per_pos=10, max_seq_num=0, input_type='fasta', verbose=True, seed=0
# )


from src.model import Model
# Load the model
model_path = "sil.model"
kwargs, model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model = Model(**kwargs)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    fake_feature = torch.rand(1, model.get_input_dim())
    fake_feature = fake_feature / fake_feature.sum(dim=1, keepdim=True)
    output = model.forward(fake_feature)


print(output.shape)
print(model.get_input_dim())