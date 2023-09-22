import os
import argparse
import json


ref_caps = json.load(open("/home/wangye/wangye2/wangye/video localization/ABIN/data/ActivityNet/test_data_audio_spk.json"))
uniq_sents = set()
asr_caps = json.load(open("/home/wangye/wangye2/wangye/TIP2021-erase/speech2text_noise3.txt"))

def main():
    for idx, data in enumerate(ref_caps):
        id, duration, timestamp, sents, sid = data[0], data[1], data[2], data[3], data[4]
        #a = sid.find("val_1")
        # clean asr
        # if sid.find("yINX46xPRf0_val_2")!=-1 or sid.find("yINX46xPRf0_val_2_1")!=-1 or sid.find("val_1")!=-1 or sid.find("train")!=-1 or sid.find('ll91M5topgU_val_2_2')!=-1 or sid.find('yBL1hCKmX7s_val_2_3')!=-1 or sid.find('lkC_md7KKq0_val_2_2')!=-1 or sid.find('lkC_md7KKq0_val_2_4')!=-1 or sid.find('lgB0Ynn38-k_val_2_1')!=-1 or sid.find('lfH_S2LTEXA_val_2_3')!=-1:
        #     continue
        # if sid[0] == 'i' and sid[1] == 'u':
        #     sid = sid[1:]


        # noise asr
        if sid.find("jz4sNglQg_val_2_3")!=-1 or sid.find("JeNdvNzhU_val_2_3")!=-1 or sid.find("ZgExTrv70_val_2_4")!=-1 or sid.find("wRwpVLE_Y_val_2_4")!=-1 or sid.find("wRwpVLE_Y_val_2_3")!=-1 or sid.find("8Sr9H3Wi8_val_2_2")!=-1 or sid.find("zufK6CufVhA_val_2_1")!=-1 or sid.find("Y5zJT3BjIxM_val_2_4")!=-1 or sid.find("fK6CufVhA_val_2_1")!=-1:
            continue
        sid = sid+'.wav'
        z = sid[1:]
        x = sid[2:]
        y = sid[3:]
        if z in asr_caps:
            translate_text = asr_caps[z]
        elif x in asr_caps:
            translate_text = asr_caps[x]
        elif y in asr_caps:
            translate_text = asr_caps[y]
        elif sid in asr_caps:
            translate_text = asr_caps[sid]
        else:
            continue
        data[3] = translate_text
        ref_caps[idx] = data
        print(idx)


    out_file = "test_data_noisy_asr.json"
    with open(out_file, 'w') as f:
        json.dump(ref_caps, f)
    print("ok")

if __name__ == '__main__':
    main()

