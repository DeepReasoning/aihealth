## Multimodal Learning in Healthcare

This repository is for the paper: **Has Multimodal Learning Delivered Universal Intelligence in Healthcare? A Comprehensive Survey**. [![arXiv](https://img.shields.io/badge/arXiv-2403.14734-b31b1b.svg)](https://arxiv.org/abs/243.14734)

#### Citation

If this work is helpful to `you`, please consider cite our paper using the following citation format:

```bibtex
@article{lin2024healthcare,
  title={Has Multimodal Learning Delivered Universal Intelligence in Healthcare? A Comprehensive Survey},
  author={Qika Lin, Yifan Zhu, Xin Mei, Ling Huang, Jingying Ma, Kai He, Zhen Peng, Erik Cambria, Mengling Feng},
  journal={arXiv preprint arXiv:2408.12880},
  year={2024}
}
```

#### News

- Update on 2024/08/23: Version 1.0 released

#### Datasets and Resoures

For Report Generation:

- IU X-ray：[[Link](https://academic.oup.com/jamia/article/23/2/304/2572395?login=true)] Preparing a collection of radiology examinations for distribution and retrieval. `2016`
- ICLEF-Caption-2017: [[Link](https://arodes.hes-so.ch/record/2258?v=pdf)] Overview of ImageCLEFcaption 2017 : image caption prediction and concept detection for biomedical images. `2017`
- ICLEF-Caption-2018: [[Link](https://repository.essex.ac.uk/22744/)] Overview of the ImageCLEF 2018 Caption Prediction Tasks. `2018`
- PEIR Gross: [[ACL](https://arxiv.org/pdf/1711.08195)] On the Automatic Generation of Medical Imaging Reports. `2018`
- ROCO: [[Link](https://academic.oup.com/jamia/article/23/2/304/2572395?login=true)] Radiology Objects in COntext (ROCO): A Multimodal Image Dataset. `2018`
- PadChest: [[Link](https://arxiv.org/pdf/1901.07441)] Padchest: A large chest x-ray image dataset with multi-label annotated reports. `2020`
- MedICaT: [[EMNLP](https://arxiv.org/pdf/2010.06000)] MedICaT: A Dataset of Medical Images, Captions, and Textual References. `2020`
- ARCH: [[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Gamper_Multiple_Instance_Captioning_Learning_Representations_From_Histopathology_Textbooks_and_Articles_CVPR_2021_paper.pdf)] Multiple Instance Captioning: Learning Representations from Histopathology Textbooks and Articles. `2021`
- FFA-IR: [[NeurIPS](https://openreview.net/pdf?id=FgYTwJbjbf)] FFA-IR: Towards an Explainable and Reliable Medical Report Generation Benchmark. `2021`
- CTRG: [[Link](https://www.sciencedirect.com/science/article/abs/pii/S0957417423019449)] Work like a doctor: Unifying scan localizer and dynamic generator for automated computed tomography report generation. `2024`

For VQA:

- VQA-Med-2018: [[Link](https://arodes.hes-so.ch/record/2780?ln=en&v=pdf)] Overview of imageCLEF 2018 medical domain visual question answering task. `2018`
- VQA-RAD: [[Link](https://www.nature.com/articles/sdata2018251)] A dataset of clinically generated visual questions and answers about radiology images. `2018`
- VQA-Med-2019: [[Link](https://arodes.hes-so.ch/record/4214?v=pdf)] VQA-Med : overview of the medical visual question answering task at ImageCLEF 2019. `2019`
- VQA-Med-2020: [[Link](https://arodes.hes-so.ch/record/6454?ln=fr&v=pdf)] Overview of the VQA-Med task at ImageCLEF 2020 : visual question answering and generation in the medical domain. `2020`
- RadVisDial-Silver: [[Link](https://aclanthology.org/2020.bionlp-1.6.pdf)] Towards Visual Dialog for Radiology. `2020`
- RadVisDial-Gold: [[Link](https://aclanthology.org/2020.bionlp-1.6.pdf)] Towards Visual Dialog for Radiology. `2020`
- PathVQA: [[Link](https://arxiv.org/pdf/2003.10286)] PathVQA: 30000+ Questions for Medical Visual Question Answering. `2020`
- VQA-Med-2021: [[Link](https://arodes.hes-so.ch/record/9062?ln=de&v=pdf)] Overview of the VQA-Med task at ImageCLEF 2021: visual question answering and generation in the medical domain. `2021`
- SLAKE: [[Link](https://arxiv.org/pdf/2102.09542)] Slake: A Semantically-Labeled Knowledge-Enhanced Dataset For Medical Visual Question Answering. `2021`
- MIMIC-Diff-VQA: [[KDD](https://dl.acm.org/doi/pdf/10.1145/3580305.3599819)] Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering. `2023`

#### Contrastive Foundatation Models

- ConVIRT: [[Link](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf)] Contrastive Learning of Medical Visual Representations
  from Paired Images and Text. `10/2020`
- PubMedCLIP: [[ACL](https://aclanthology.org/2023.findings-eacl.88.pdf)] PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? `12/2021`
- CheXzero: [[Nature Biomedical Engineering](https://www.nature.com/articles/s41551-022-00936-9)] Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning `09/2022`
- BiomedCLIP: [[Link](https://arxiv.org/pdf/2303.00915)] BiomedCLIP: a multimodal biomedical foundation model
  pretrained from fifteen million scientific image-text pairs. `03/2023`
- PLIP: [[Nature Medicine](https://www.nature.com/articles/s41591-023-02504-3)] A visual–language foundation model for pathology image analysis using medical Twitter. `03/2023`
- PathCLIP: [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28308)] PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology. `05/2023`
- CT-CLIP: [[Link](https://arxiv.org/pdf/2403.17834)] A foundation model utilizing chest CT volumes and radiology
  reports for supervised-level zero-shot detection of abnormalities. `03/2024`
- PairAug: [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_PairAug_What_Can_Augmented_Image-Text_Pairs_Do_for_Radiology_CVPR_2024_paper.pdf)] PairAug: What Can Augmented Image-Text Pairs Do for Radiology? `04/2023`

- GLoRIA: [[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)] GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition. `10/2021`
- BioViL: [[ECCV](https://arxiv.org/pdf/2204.09817)] Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing. `04/2022`
- MedCLIP: [[EMNLP](https://arxiv.org/pdf/2210.10163)] MedCLIP: Contrastive Learning from Unpaired Medical Images and Text. `10/2022`
- MGCA: [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/d925bda407ada0df3190df323a212661-Paper-Conference.pdf)] Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning. `10/2022`
- BioViL-T: [[CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.pdf)] Learning to Exploit Temporal Structure for Biomedical Vision–Language Processing. `01/2023`
- MedKLIP: [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_MedKLIP_Medical_Knowledge_Enhanced_Language-Image_Pre-Training_for_X-ray_Diagnosis_ICCV_2023_paper.pdf)] MedKLIP: Medical Knowledge Enhanced Language-Image
  Pre-Training for X-ray Diagnosis. `01/2023`
- KAD: [[Nature Communications](https://www.nature.com/articles/s41467-023-40260-7)] Knowledge-enhanced visual-language pre-training on chest radiology images. `2/2023`
- PTUnifier: [[ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Towards_Unifying_Medical_Vision-and-Language_Pre-Training_via_Soft_Prompts_ICCV_2023_paper.pdf)] Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts. `02/2023`
- Med-UniC: [[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/af38fb8e90d586f209235c94119ba193-Paper-Conference.pdf)] Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias. `05/2023`
- MCR: [[Link](https://arxiv.org/pdf/2312.15840)] Masked Contrastive Reconstruction for Cross-modal Medical Image-Report Retrieval. `12/2023`
- MLIP: [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MLIP_Enhancing_Medical_Visual_Representation_with_Divergence_Encoder_and_Knowledge-guided_CVPR_2024_paper.pdf)] MLIP: Enhancing Medical Visual Representation with Divergence Encoder and Knowledge-guided Contrastive Learning. `02/2024`
- MAVL: [[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Phan_Decomposing_Disease_Descriptions_for_Enhanced_Pathology_Detection_A_Multi-Aspect_Vision-Language_CVPR_2024_paper.pdf)] Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Pre-training Framework. `03/2024`
- KEP: [[Link](https://arxiv.org/pdf/2404.09942)] Knowledge-enhanced Visual-Language Pretraining for Computational Pathology. `04/2024`
- DeViDe: [[Link](https://arxiv.org/pdf/2404.03618)] DeViDe: Faceted medical knowledge for improved medical vision-language pre-training. `04/2024`

#### Multimodal Large Language Models

- SkinGPT-4: [[Nature Communications](https://www.nature.com/articles/s41467-024-50043-3)] Pre-trained multimodal large language model enhances dermatological diagnosis using SkinGPT-4. `04/2023`
- PathAsst: [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28308)] PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology. `05/2023`
- MedBLIP: [[Link](https://arxiv.org/pdf/2305.10799)] MedBLIP: Bootstrapping Language-Image Pre-training from 3D Medical Images and Texts. `05/2023`
- LLM-CXR: [[ICLR](https://arxiv.org/pdf/2305.11490)] LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation. ``05/2023`
- BiomedGPT: [[Nature Medicine](https://www.nature.com/articles/s41591-024-03185-2)] A generalist vision–language foundation model for diverse biomedical tasks. `05/2023`
- XrayGPT: [[Link](https://arxiv.org/pdf/2306.07971)] XrayGPT: Chest Radiographs Summarization using Large Medical
  Vision-Language Models. `06/2023`
- LLaVA-Med: [[Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf)] LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. `06/2023`
- Med-Flamingo: [[Link](https://proceedings.mlr.press/v225/moor23a/moor23a.pdf)] Med-Flamingo: a Multimodal Medical Few-shot Learner. `07/2023`
- Med-PaLM M: [[Link](https://arxiv.org/pdf/2307.14334)] Towards Generalist Biomedical AI. `07/2023`
- RadFM: [[Link](https://arxiv.org/pdf/2308.02463)] Towards Generalist Foundation Model for Radiology by
  Leveraging Web-scale 2D&3D Medical Data. `08/2023`
- RaDialog: [[Link](https://arxiv.org/pdf/2311.18681)] RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance. `10/2023`
- Qilin-Med-VL: [[Link](https://arxiv.org/pdf/2310.17956)] Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare. `10/2023`
- MAIRA-1: [[Link](https://arxiv.org/pdf/2311.13668)] Maira-1: A specialised large multimodal model
  for radiology report generation. `11/2023`
- PathChat: [[Link](https://arxiv.org/pdf/2312.07814)] A Foundational Multimodal Vision Language AI Assistant for
  Human Pathology. `12/2023`
- MedXChat: [[Link](https://arxiv.org/pdf/2312.02233)] MedXChat: A Unified Multimodal Large Language Model Framework towards CXRs Understanding and Generation. `12/2023`
- CheXagent: [[Link](https://arxiv.org/pdf/2401.12208)] CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation. `01/2024`
- CONCH: [[Nature Medicine](https://www.nature.com/articles/s41591-024-02856-4)] A visual-language foundation model for computational pathology. `03/2024`
- M3D-LaMed: [[Link](https://arxiv.org/pdf/2404.00578)] M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models. `03/2024`
- Dia-LLaMA: [[Link](https://arxiv.org/pdf/2403.16386)] Dia-LLaMA: Towards Large Language Model-driven CT Report Generation. `03/2024`
- LLaVA-Rad: [[Link](https://arxiv.org/pdf/2403.08002)] Towards a clinically accessible radiology multimodal
  model: open-access and lightweight, with automatic evaluation. `03/2024`
- WoLF: [[Link](https://arxiv.org/pdf/2403.15456)] WoLF: Wide-scope Large Language Model Framework for CXR Understanding. `03/2024`
