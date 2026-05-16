from datasets import load_dataset

print("Checking AppTek...")
try:
    ds_apptek = load_dataset("apptek-com/apptek_callcenter_dialogues", split="train", streaming=True)
    sample = next(iter(ds_apptek))
    print("AppTek sample keys:", sample.keys())
    # print some metadata if possible
    # print("AppTek speaker info:", sample.get("speaker_info", "N/A"))
except Exception as e:
    print("AppTek Error:", e)

print("\nChecking CommonPhone...")
try:
    ds_cp = load_dataset("pklumpp/CommonPhoneDataset", "en", split="train", streaming=True)
    sample = next(iter(ds_cp))
    print("CommonPhone sample keys:", sample.keys())
except Exception as e:
    print("CommonPhone Error:", e)
