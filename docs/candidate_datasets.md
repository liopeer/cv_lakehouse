<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# Candidate datasets

These datasets fit the PII use case, and the lake does not hold them yet. A licence that
forbids commercial use does not exclude a dataset. `commercial_use` records it, and a
consumer applies its own rule.

"Unverified" means that no official page states the licence. Check it before you add the
dataset.

## Faces

| Dataset | Classes | Size | Licence | Distribution |
| --- | --- | --- | --- | --- |
| [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/) | face, at night and in low light | 6,000 labelled images | none stated | Google Drive, Baidu |
| [UFDD](https://ufdd.info/) | face, in rain, snow, haze, blur and lens obstruction | about 6,400 images | none stated | Google Drive, Baidu |
| MAFA | face, masked or occluded | 30,811 images, 35,806 faces | unverified | no official page found |
| [ImageNet face annotations](https://github.com/princetonvisualai/imagenet-face-obfuscation) | face | about 562k faces on 243k ILSVRC images | BSD-2 annotations, ImageNet images after a login | annotations direct, images gated |

## Licence plates

| Dataset | Classes | Size | Licence | Distribution |
| --- | --- | --- | --- | --- |
| [UC3M-LP](https://github.com/ramajoballester/UC3M-LP) | license_plate, Spanish plates in the EU format | 1,975 images, 2,547 plates | ODbL. Commercial use needs the consent of the authors. | one zip on [Dataverse](https://doi.org/10.21950/OS5W4Z), with an md5 |
| [CCPD2019 and CCPD-Green](https://github.com/detectRecog/CCPD) | license_plate, one Chinese plate per image, with corners and text | about 300k images | MIT | Google Drive, Baidu |
| [CRPD](https://github.com/yxgong0/CRPD) | license_plate, several Chinese plates per image | about 34.5k images | no licence in the repository | Google Drive, Baidu |
| [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) | license_plate, Brazilian plates | 20,000 images | non-commercial research | a signed agreement by email |
| [UFPR-ALPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) | license_plate, Brazilian plates | 4,500 images | non-commercial research | a signed agreement by email |

## Heads and persons

| Dataset | Classes | Size | Licence | Distribution |
| --- | --- | --- | --- | --- |
| [CrowdHuman](https://www.crowdhuman.org/) | head, person (visible box and full box) | 15k train, 4.4k val, 5k test. About 23 persons per image. | non-commercial research | Google Drive, Baidu, 7 files |
| [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) | head, in classrooms and from the web | 4,405 images, 111k heads | academic research | Google Drive, Baidu |
| [HT21](https://motchallenge.net/data/Head_Tracking_21/) | head, tracked in dense crowds | 11,463 frames, 2.28M heads | CC-BY-NC-SA-3.0 for the site, unverified for HT21 | direct zips |
| [CEPDOF](https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/) | person from above, through a fisheye lens, as rotated boxes | about 25k frames | ODbL and DbCL | one direct zip of 9 GB |
| [WEPDTOF](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/) | person from above, through a fisheye lens | 16 videos | CC-BY-NC-4.0 | one direct zip |
| [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/) | person, pedestrians in five classes | 13,382 images | none stated | Google Drive, Baidu |
| [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | person, from a drone | 10,209 images and videos | unverified | Google Drive, Baidu |
| [CityPersons](https://www.cityscapes-dataset.com/) | person, pedestrians on the road | the Cityscapes images | non-commercial | a registration |

## Text

| Dataset | Classes | Size | Licence | Distribution |
| --- | --- | --- | --- | --- |
| [COCO-Text v2](https://vision.cornell.edu/se3/coco-text-2/) | text, on COCO images, with legible and handwritten flags | about 63k images | CC-BY-4.0 annotations, unverified | one zip on GitHub. It names the COCO 2014 files, which hold the same images as `bronze/coco`. |
| [Total-Text](https://github.com/cs-chan/Total-Text-Dataset) | text, curved | 1,555 images | non-commercial research | GitHub |
| [ICDAR MLT-2019, ArT, IC15](https://rrc.cvc.uab.es/) | text | various | unverified | a registration |

## Before you add a dataset

Each dataset labels only some classes. COCO labels persons and no faces, and TextOCR
labels text and no faces. A model that trains on two of them learns an unlabelled face
as background. Before gold mixes them, each dataset has to state the classes that it
labels completely.

Some datasets publish this per image: LVIS lists the categories that are not labelled
completely on each image, and Open Images lists the labels that annotators verified as
present or absent.

## Avoid

| Dataset | Reason |
| --- | --- |
| SA-1B, Mapillary Vistas, nuScenes, Argoverse, Ego4D, Cityscapes blurred | Faces and plates are blurred in the images. As background, they teach a model that a blurred face is no face. |
| Brainwash | Stanford withdrew it. |
| Uber-Text | The official page is gone. |
| Roboflow and Kaggle plate sets | The origin of the images is unknown. |
| CelebA, FFHQ, FDF | Face crops, not detection data. |
