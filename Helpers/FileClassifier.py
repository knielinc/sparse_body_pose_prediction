import re
from enum import Enum
import os
class move_type(Enum):
    WALKING = 1
    RUNNING = 2
    CROUCHING = 3
    STANDING = 4
    THROWING = 5
    BOXING = 6
    INTERACTION = 7 #(object, gestures)
    DANCING = 8
    WORKOUT = 9
    BASKETBALL = 10
    RACKETSPORT = 11 #Badminton / Tennis
    GOLF = 12
    RANGEOFMOTION = 13
    UNKNOWN = 14

    @staticmethod
    def type_as_string(move_type):
        return str(move_type).split('.')[1]


class mocap_dataset(Enum):
    ACCAD = 1
    BMLMOVI = 2
    BMLRUB = 3
    BMLHANDBALL = 4
    CMU = 5
    DFAUST = 6
    EKUT = 7
    EYESJPN = 8
    HUMANEVA = 9
    KIT = 10
    MPIHDM05 = 11
    POSEPRIOR = 12
    MPIMOSH = 13
    SFU = 14
    SSM = 15
    TCDHANDS = 16
    TOTALCAPTURE = 17
    TRANSITIONS = 18

UNKNOWN_KEYWORDS = ["egyptian", "accident", "treadmill", "motorcycle", "martial arts", "ladder", "stair", "flip",
                    "roll", "hop on", "playground", "uneven", "lie", "sway", "stair", "beam", "parcour", "vault"
                    "swing", "cal", "walkdog", "cartwheel", "kick", "sit", "jump", "climb", "resists", "moonwalk",
                    "human subject", "crawl", "gus", "joof"]
RUNNING_KEYWORDS = ["run", "jog", "side_step"]
WALKING_KEYWORDS = ["walk", "CleanedGRS", "march", "navigate", "turn", "step"]
CROUCHING_KEYWORDS = ["crouch"]
STANDING_KEYWORDS = ["stand", "look", "wait"]
THROWING_KEYWORDS = ["throw", "catch", "frisbee"]
BOXING_KEYWORDS = ["punch", "attack", "boxing", "lunge"]
INTERACTION_KEYWORDS = ["acting", "knock", "lift", "pick up", "clean", "hair", "chopping", "digging", "hand", "gun", "vacuuming",
                        "chop", "sewing", "planting", "fishing", "phone", "water", "sipping", "eating", "carry",
                        "gesture", "laugh", "cry", "drink", "wash", "mop", "sweep", "hand", "shake", "high-five",
                        "low-five", "signal", "vignettes", "wrench", "saw", "screw", "hammer", "cleaning", "open",
                        "close", "bolt", "placing", "picking", "greeting", "wipe"]
DANCING_KEYWORDS = ["dance", "salsa", "chacha"]
WORKOUT_KEYWORDS = ["jacks", "squats", "stretch", "flexing", "weight lift", "bend"]
BASKETBALL_KEYWORDS = ["basketball"]
RACKETSPORT_KEYWORDS = ["baseball", "tennis", "badminton"]
GOLF_KEYWORDS = ["golf", "swing", "put"]
RANGEOFMOTION_KEYWORDS = ["rom", "range", "arms", "shoulders", "knee"]

ALL_KEYWORD_LISTS = [(move_type.UNKNOWN,        UNKNOWN_KEYWORDS),
                     (move_type.CROUCHING,      CROUCHING_KEYWORDS),
                     (move_type.STANDING,       STANDING_KEYWORDS),
                     (move_type.THROWING,       THROWING_KEYWORDS),
                     (move_type.BOXING,         BOXING_KEYWORDS),
                     (move_type.INTERACTION,    INTERACTION_KEYWORDS),
                     (move_type.DANCING,        DANCING_KEYWORDS),
                     (move_type.WORKOUT,        WORKOUT_KEYWORDS),
                     (move_type.BASKETBALL,     BASKETBALL_KEYWORDS),
                     (move_type.RACKETSPORT,    RACKETSPORT_KEYWORDS),
                     (move_type.GOLF,           GOLF_KEYWORDS),
                     (move_type.RUNNING,        RUNNING_KEYWORDS),
                     (move_type.WALKING,        WALKING_KEYWORDS),
                     (move_type.RANGEOFMOTION,  RANGEOFMOTION_KEYWORDS)]


def get_path_tuples(src_path):
    ACCAD_PATH          = src_path + "\\ACCAD"
    BMLRUB_PATH         = src_path + "\\BioMotionLab_NTroje"
    BMLHANDBALL_PATH    = src_path + "\\BMLhandball"
    BMLMOVI_PATH        = src_path + "\\BMLmovi"
    CMU_PATH            = src_path + "\\CMU"
    DFAUST_PATH         = src_path + "\\DFaust_67"
    EKUT_PATH           = src_path + "\\EKUT"
    EYESJPN_PATH        = src_path + "\\Eyes_Japan_Dataset"
    HUMANEVA_PATH       = src_path + "\\HumanEva"
    KIT_PATH            = src_path + "\\KIT"
    MPIHDM05_PATH       = src_path + "\\MPI_HDM05"
    MPILIMITS_PATH      = src_path + "\\MPI_Limits"
    MPIMOSH_PATH        = src_path + "\\MPI_mosh"
    SFU_PATH            = src_path + "\\SFU"
    SSM_PATH            = src_path + "\\SSM_synced"
    TCDHANDS_PATH       = src_path + "\\TCD_handMocap"
    TOTALCAPTURE_PATH   = src_path + "\\TotalCapture"
    TRANS_PATH          = src_path + "\\Transitions_mocap"
    
    PATH_TUPLES = [(ACCAD_PATH,         mocap_dataset.ACCAD),
                   (BMLRUB_PATH,        mocap_dataset.BMLRUB),
                   (BMLHANDBALL_PATH,   mocap_dataset.BMLHANDBALL),
                   (BMLMOVI_PATH,       mocap_dataset.BMLHANDBALL),
                   (CMU_PATH,           mocap_dataset.CMU),
                   (DFAUST_PATH,        mocap_dataset.DFAUST),
                   (EKUT_PATH,          mocap_dataset.EKUT),
                   (EYESJPN_PATH,       mocap_dataset.EYESJPN),
                   (HUMANEVA_PATH,      mocap_dataset.HUMANEVA),
                   (KIT_PATH,           mocap_dataset.KIT),
                   (MPIHDM05_PATH,      mocap_dataset.MPIHDM05),
                   (MPILIMITS_PATH,     mocap_dataset.POSEPRIOR),
                   (MPIMOSH_PATH,       mocap_dataset.MPIMOSH),
                   (SFU_PATH,           mocap_dataset.SFU),
                   (SSM_PATH,           mocap_dataset.SSM),
                   (TCDHANDS_PATH,      mocap_dataset.TCDHANDS),
                   (TOTALCAPTURE_PATH,  mocap_dataset.TOTALCAPTURE),
                   (TRANS_PATH,         mocap_dataset.TRANSITIONS)]
    
    return PATH_TUPLES

CMU_MAPPING = None

def classify_by_name(file_name):
    for tuple in ALL_KEYWORD_LISTS:

        curr_move_type = tuple[0]
        keyword_list = tuple[1]

        for keyword in keyword_list:
            if re.search(keyword, file_name, re.IGNORECASE):
                return  curr_move_type

    return move_type.UNKNOWN

def classifyACCAD(folder_name, file_name):
    if re.search("martial", folder_name, re.IGNORECASE) and not re.search("kick", folder_name, re.IGNORECASE):
        return move_type.BOXING

    return classify_by_name(file_name)

def classifyBMLMOVI(file_name):
    idx = int(file_name.split('_')[-2])

    if idx == 1:
        return move_type.WALKING #walking
    if idx == 2:
        return move_type.RUNNING #jogging
    if idx == 3:
        return move_type.UNKNOWN #running in place
    if idx == 4:
        return move_type.RUNNING #side gallop
    if idx == 5:
        return move_type.UNKNOWN #crawling
    if idx == 6:
        return move_type.UNKNOWN #Vertical Jumping
    if idx == 7:
        return move_type.UNKNOWN #Jumping Jacks
    if idx == 8:
        return move_type.UNKNOWN #Kicking
    if idx == 9:
        return move_type.WORKOUT #Streching
    if idx == 10:
        return move_type.UNKNOWN #Crossarms
    if idx == 11:
        return move_type.UNKNOWN #Sitting
    if idx == 12:
        return move_type.UNKNOWN #crossed sitting
    if idx == 13:
        return move_type.INTERACTION #pointing
    if idx == 14:
        return move_type.INTERACTION #clapping
    if idx == 15:
        return move_type.INTERACTION #headscratching
    if idx == 16:
        return move_type.THROWING #throwing and catching
    if idx == 17:
        return move_type.INTERACTION #waving
    if idx == 18:
        return move_type.INTERACTION #pretending to take picture
    if idx == 19:
        return move_type.INTERACTION #pretending to check watch

    return move_type.UNKNOWN

def classifyBMLRUB(folder_name, file_name):
    if re.search("rub", folder_name, re.IGNORECASE): #walking in place and not on ground
        if re.search("normal_jog", file_name, re.IGNORECASE) or re.search("normal__walk", file_name, re.IGNORECASE):
            return move_type.UNKNOWN

    return classify_by_name(file_name)

def init_cmu_mapping():
    global CMU_MAPPING
    CMU_MAPPING = dict()
    with open(os.getcwd() + '\\CMUMapping.txt', 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            splitted = line.split('\t')
            key = splitted[0]
            value = ""
            if splitted.__len__() > 1:
                value = splitted[1]
            else:
                value = "failed"
            CMU_MAPPING[key] = value

def classifyCMU(folder_name, file_name):
    if CMU_MAPPING == None:
        init_cmu_mapping()

    key = file_name.split("_poses")[0]
    if key in CMU_MAPPING.keys():
        value = CMU_MAPPING[key]
        return classify_by_name(value)
    return move_type.UNKNOWN

def classifyEYESJPN(file_name):
    return classify_by_name(file_name)

def classifyKIT(file_name):
    return classify_by_name(file_name)

def classifyMPIHDM05(file_name):
    key = file_name.split("_120_poses")[0].split("_")[2]

    HDM05_Map = {
        "01-01" : move_type.WALKING, #walking
        "01-02" : move_type.WALKING, #locomotion on spot (no treadmill was used so i keep the data)
        "01-03" : move_type.UNKNOWN, #locomotion
        "01-04" : move_type.WALKING, #Locomotion with weights
        "02-01" : move_type.INTERACTION, #Table and floor objects
        "02-02" : move_type.INTERACTION, #Shelf
        "02-03" : move_type.INTERACTION, #Shelf
        "03-01" : move_type.DANCING, #Dancing
        "03-02" : move_type.UNKNOWN, #Kicking and punching
        "03-03" : move_type.THROWING, #Throwing
        "03-04" : move_type.WORKOUT, #Rotating arms
        "03-05" : move_type.WORKOUT, #Workout
        "03-06" : move_type.WORKOUT, #Workout
        "03-07" : move_type.WORKOUT, #Workout
        "03-08" : move_type.UNKNOWN, #Rope skipping
        "03-09" : move_type.RACKETSPORT, #Badminton
        "03-10" : move_type.UNKNOWN, #Sitting and Lying Down
        "03-11" : move_type.UNKNOWN, #Chair, table, floor
        "04-01" : move_type.INTERACTION, #Miscellaneous Motions
        "05-01" : move_type.INTERACTION, #Clapping and waving
        "05-02" : move_type.INTERACTION, #Shouting and tying shoes
        "05-03" : move_type.UNKNOWN, #variations of locomotion
    }
    if key in HDM05_Map.keys():
        return HDM05_Map[key]
    else:
        return move_type.UNKNOWN

def classifyMPIMOSH(file_name):
    return classify_by_name(file_name)

def classifySFU(file_name):
    return classify_by_name(file_name)

def classifySSM(file_name):
    return classify_by_name(file_name)

def classifyTOTALCAPTURE(file_name):
    return classify_by_name(file_name)

def classifyHUMANEVA(file_name):
    return classify_by_name(file_name)

def classifyTRANSITIONS(file_name):
    return  classify_by_name(file_name)

def classify_file(mocap_dataset_class, folder_name, file_name):
    if mocap_dataset_class == mocap_dataset.ACCAD:
        return classifyACCAD(folder_name, file_name)
    if mocap_dataset_class == mocap_dataset.BMLMOVI:
        return classifyBMLMOVI(file_name)
    if mocap_dataset_class == mocap_dataset.BMLRUB:
        return classifyBMLRUB(folder_name, file_name)
    if mocap_dataset_class == mocap_dataset.BMLHANDBALL:
        return move_type.THROWING
    if mocap_dataset_class == mocap_dataset.CMU:
        return classifyCMU(folder_name, file_name)
    if mocap_dataset_class == mocap_dataset.DFAUST:
        return move_type.UNKNOWN #Honestly not many usable animations, mostly jiggle or hopping to demostrate tissue motion, not usable for vr
    if mocap_dataset_class == mocap_dataset.EKUT:
        #TODO FIND WAY TO CLASSIFY
        return move_type.UNKNOWN
    if mocap_dataset_class == mocap_dataset.EYESJPN:
        return classifyEYESJPN(file_name)
    if mocap_dataset_class == mocap_dataset.KIT:
        return classifyKIT(file_name)
    if mocap_dataset_class == mocap_dataset.MPIHDM05:
        return  classifyMPIHDM05(file_name)
    if mocap_dataset_class == mocap_dataset.POSEPRIOR:
        return move_type.RANGEOFMOTION
    if mocap_dataset_class == mocap_dataset.MPIMOSH:
        return classifyMPIMOSH(file_name)
    if mocap_dataset_class == mocap_dataset.SFU:
        return classifySFU(file_name)
    if mocap_dataset_class == mocap_dataset.SSM:
        return classifySSM(file_name)
    if mocap_dataset_class == mocap_dataset.TCDHANDS:
        return move_type.UNKNOWN #all poses are in sitting positions (not stable, when standing)
    if mocap_dataset_class == mocap_dataset.TOTALCAPTURE:
        return classifyTOTALCAPTURE(file_name)
    if mocap_dataset_class == mocap_dataset.HUMANEVA:
        return classifyHUMANEVA(file_name)
    if mocap_dataset_class == mocap_dataset.TRANSITIONS:
        return classifyTRANSITIONS(file_name)

    pass