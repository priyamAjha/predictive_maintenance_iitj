import pandas as pd

index_names = ['engine', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names=["Fan_inlet_temperature_R",
"LPC_outlet_temperature_R",
"HPC_outlet_temperature_R",
"LPT_outlet_temperature_R",
"Fan_inlet_Pressure_psia",
"bypass_duct_pressure_psia",
"HPC_outlet_pressure_psia",
"Physical_fan_speed_rpm",
"Physical_core_speed_rpm",
"Engine_pressure_ratioP50_P2",
"HPC_outlet_Static_pressure_psia",
"Ratio_of_fuel_flow_to_Ps30_pps_psia",
"Corrected_fan_speed_rpm",
"Corrected_core_speed_rpm",
"Bypass_Ratio",
"Burner_fuel_air_ratio",
"Bleed_Enthalpy",
"Required_fan_speed",
"Required_fan_conversion_speed",
"High_pressure_turbines_Cool_air_flow",
"Low_pressure_turbines_Cool_air_flow"]
trainTestColumns = index_names + setting_names + sensor_names

def loadData():
    dirPath = '../data/'
    trainTxtFiles = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']
    testTxtFiles = ['test_FD001.txt', 'test_FD002.txt', 'test_FD003.txt', 'test_FD004.txt']
    rulTxtFiles = ['RUL_FD001.txt', 'RUL_FD002.txt', 'RUL_FD003.txt', 'RUL_FD004.txt']
    trueRulTxtFile = ['x.txt']    
    trainDatasets = []
    testDatasets = []
    expectedRulDatasets = []
    
    for i in range(4):
        # Import Files
        tempTrain = pd.read_csv( dirPath + trainTxtFiles[i], sep = " ", header = None)
        tempTest = pd.read_csv( dirPath + testTxtFiles[i], sep = " ", header = None)
        tempRul = pd.read_csv( dirPath + rulTxtFiles[i], sep = " ", header = None)

        # Cleaning Files
        tempTrain.drop(inplace = True, columns = [26, 27])
        tempTest.drop(inplace = True, columns = [26, 27])
        tempRul.drop(inplace = True, columns = [1])

        # Adding Columns Names
        tempTrain.columns = tempTest.columns = trainTestColumns
        tempRul.columns = ["Expected RUL"]

        # Appending to Lists
        trainDatasets.append(tempTrain)
        testDatasets.append(tempTest)
        expectedRulDatasets.append(tempRul)
    
    return trainDatasets, testDatasets, expectedRulDatasets

