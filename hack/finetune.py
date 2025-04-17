import os
import xml.etree.ElementTree as ET

classes = [ 'rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer',
    'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper',
    'rice water weevil', 'rice leafhopper', 'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket',
    'wireworm', 'white margined moth', 'black cutworm', 'large cutworm', 'yellow cutworm', 'red spider', 'corn borer',
    'army worm', 'aphids', 'Potosiabre vitarsis', 'peach borer', 'english grain aphid', 'green bug',
    'bird cherry-oataphid', 'wheat blossom midge', 'penthaleus major', 'longlegged spider mite',
    'wheat phloeothrips', 'wheat sawfly', 'cerodonta denticornis', 'beet fly', 'flea beetle', 'cabbage army worm',
    'beet army worm', 'Beet spot flies', 'meadow moth', 'beet weevil', 'sericaorient alismots chulsky', 'alfalfa weevil',
    'flax budworm', 'alfalfa plant bug', 'tarnished plant bug', 'Locustoidea', 'lytta polita', 'legume blister beetle',
    'blister beetle', 'therioaphis maculata Buckton', 'odontothrips loti', 'Thrips', 'alfalfa seed chalcid',
    'Pieris canidia', 'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 'Colomerus vitis',
    'Brevipoalpus lewisi McGregor', 'oides decempunctata', 'Polyphagotars onemus latus', 'Pseudococcus comstocki Kuwana',
    'parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus', 'Cicadella viridis', 'Miridae',
    'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 'Panonchus citri McGregor',
    'Phyllocoptes oleiverus ashmead', 'Icerya purchasi Maskell', 'Unaspis yanonensis', 'Ceroplastes rubens',
    'Chrysomphalus aonidum', 'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus',
    'Tetradacus c Bactrocera minax', 'Dacus dorsalis(Hendel)', 'Bactrocera tsuneonis', 'Prodenia litura',
    'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii',
    'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar',
    'Salurnis marginella Guerr', 'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper',
    'Rhytidodera bowrinii white', 'Sternochetus frigidus', 'Cicadellidae'
]

def convert_annotation(xml_folder, save_folder, img_folder):
    os.makedirs(save_folder, exist_ok=True)
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)

        label_path = os.path.join(save_folder, xml_file.replace('.xml', '.txt'))
        with open(label_path, 'w') as out_file:
            for obj in root.findall('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)

                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text)
                xmax = float(xml_box.find('xmax').text)
                ymin = float(xml_box.find('ymin').text)
                ymax = float(xml_box.find('ymax').text)

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                out_file.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

# def convert_annotation(xml_folder, save_folder, img_folder):
#     os.makedirs(save_folder, exist_ok=True)
#     for xml_file in os.listdir(xml_folder):
#         if not xml_file.endswith('.xml'):
#             continue
#         full_path = os.path.join(xml_folder, xml_file)
#         try:
#             tree = ET.parse(full_path)
#             root = tree.getroot()
#         except ET.ParseError as e:
#             print(f"Skipping invalid XML: {xml_file} -> {e}")
#             continue



import os
import xml.etree.ElementTree as ET

def clean_invalid_xml(xml_folder):
    removed_count = 0
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        full_path = os.path.join(xml_folder, xml_file)
        try:
            ET.parse(full_path)
        except ET.ParseError as e:
            print(f"Deleting invalid XML -> {xml_file}: {e}")
            os.remove(full_path)
            removed_count += 1
    print(f"\nCleanup complete. {removed_count} invalid file(s) deleted.")

# Usage:
# clean_invalid_xml('Dataset/Annotations')





# Example:
convert_annotation('Dataset/Annotations', 'Dataset/labels', 'Dataset/JPEGImages')
