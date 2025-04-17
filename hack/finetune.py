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
    
    # Track all classes found for reference
    found_classes = set()
    
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
            try:
                objects = root.findall('object')
                print(f"Found {len(objects)} objects in {xml_file}")
                if len(objects) == 0:
                    print(f"No objects found in {xml_file}")
                
                for obj in objects:
                    cls = obj.find('name').text
                    print(f"Object class: {cls}")
                    found_classes.add(cls)
                    
                    # Try to convert the class to an integer (index in the classes list)
                    # If that fails, use it as a direct index
                    try:
                        # Convert numeric class ID to integer
                        class_index = int(cls)
                        
                        # Check if this index is within the range of our classes list
                        if 0 <= class_index < len(classes):
                            # Use the class index directly in YOLO format
                            cls_id = class_index
                        else:
                            # If it's out of range, we'll just use the number as is
                            cls_id = class_index
                    except ValueError:
                        # If it's not a number, check if it's in our classes list
                        if cls in classes:
                            cls_id = classes.index(cls)
                        else:
                            print(f"Unknown class '{cls}', skipping")
                            continue
                    
                    xml_box = obj.find('bndbox')
                    xmin = float(xml_box.find('xmin').text)
                    xmax = float(xml_box.find('xmax').text)
                    ymin = float(xml_box.find('ymin').text)
                    ymax = float(xml_box.find('ymax').text)

                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    width = (xmax - xmin) / w
                    height = (ymax - ymin) / h

                    print(f"Writing to file: {cls_id} {x_center} {y_center} {width} {height}")
                    out_file.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                    
            except Exception as e:  
                print(f"Error processing {xml_file}: {e}")
    
    # Print all unique classes found for reference
    print(f"All classes found: {sorted(list(found_classes))}")
    return found_classes

# Example usage:
convert_annotation('Dataset/Annotations', 'Dataset/labels', 'Dataset/JPEGImages')