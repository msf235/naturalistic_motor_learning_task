import mlcroissant as mlc
import numpy as np
import sys

ds = mlc.Dataset("./metadata.json")
records = ds.records(record_set="phase-2")
for record in records:
    print()
    print(record)
sys.exit()
breakpoint()

# # ds = mlc.Dataset("./metadata2.json")
# # records = ds.records(record_set="phase-1/ctrls")
# # records = ds.records(record_set="phase-1/state")
# # ds = mlc.Dataset("./metadata3.json")
# ds = mlc.Dataset("./metadata4.json")
# records = ds.records(record_set="phase-1-state-set")
# # records = ds.records(record_set="phase-1-state-set")
# records_it = iter(records)
# print()
# print(next(records_it))
# print(next(records_it))
# print()
# sys.exit()
# # records = ds.records(record_set="phase-1/state")
# # records_it = iter(records)
# # print(next(records_it))
# # print(next(records_it))
# ds = mlc.Dataset("./flores-200-2.json")
# records = ds.records(record_set="language_translations_train_data_with_metadata")
# records_it = iter(records)
# print()
# print(next(records_it))
# # print(next(records_it))
# print()
