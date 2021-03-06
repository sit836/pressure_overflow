IN_PATH = "D:/py_projects/pressure_overflow/data/"
OUT_PATH = "D:/py_projects/pressure_overflow/output/"

TARGET = "Downhole Gauge Pressure"
DATE = "date"
WELL_NUMBER = "Well_Number"
FLOWRATE = "Tubing Flow Meter"
ELAPSED_DAYS = "elapsed_days"

COLS_AUX = ["date", "Well_Number"]
COLS_CAT = [
    "Downhole Gauge Type",
    "Tubing Flow Meter Scale High|Meter Type",
    "Well_Operator_Run",
    "Operating_Area_Name",
]
COLS_NUM_TO_ENC = ['Pump Torque', 'FCV Position Feedback', 'Water Gathering Pressure',
                   'Tubing Flow Meter', 'WEC PCP Efficiency', 'Pump Speed Actual',
                   'Gas Gathering Pressure', 'Casing Pressure', 'Gas Flow (Energy)',
                   'Tubing Pressure', 'WEC PCP Theoretical Pump Displacement',
                   'Casing Pressure Gradient', 'Pump Speed Actual Min',
                   'Pump Speed Actual Max', 'Water Flow Mag from Separator',
                   'Separator Gas Pressure']
COLS_NUM_NO_ENC = ['pump_bottom_depth',
                   'sensor_depth',
                   'Well_Surface_Latitude',
                   'Well_Surface_Longitude',
                   'GL_MKB',
                   ]
COLS_USE = COLS_NUM_TO_ENC + COLS_NUM_NO_ENC + COLS_CAT + COLS_AUX

CAT_FEA_DICT = {
                'Downhole Gauge Type': {'f3a8d467eb09c3c67011d0e27f80ffaf': 0, '446aa106c277fcea4c5782f911c55e9a': 1,
                                        '3edf8ca26a1ec14dd6e91dd277ae1de6': 2, '6fbb7291e0e5b2a5553aeabd598d0eb3': 3},
                'Tubing Flow Meter Scale High|Meter Type': {'Magflow 4W-DN25': 0, 'Low Flow Wedge': 1,
                                                            'High Flow Wedge': 2, 'Not Selected': 3,
                                                            'Magflow 4W-DN40': 4, 'Small CL900 Orifice Plate': 5,
                                                            'RDS Orifice plate': 6, 'Medium CL900 Orifice Plate': 7,
                                                            'Magflow 2W-DN40': 8},
                'Well_Operator_Run': {'465d1bd5a8de030984c2a95b5617be50': 0, 'eabf47828cec6d7684b01516164cd412': 1,
                                      '19fdac452fad4f942bef8eb3ec65531c': 2, '717bbdc480419463a93b6694b71dcac4': 3,
                                      'c67519eed7c1b4cb78811d6c0ee96861': 4, 'ac0a695af4275d1a43f5c7a7c1c6b4a3': 5,
                                      '1283e4b938d82e030fdb0e19b68d606a': 6, '2943b7489df5c2064e4d3d4adbd2f00a': 7,
                                      'bb21f204f0211bbfc6fa9e4df80e3840': 8, '5377ec2060b850513b0de4da37636795': 9,
                                      '5f7964ad7fdcc1192020470da60f4537': 10, '1d9b6410e51f1b03bb8b57c0bba86214': 11,
                                      'dc47d8c14800aaf4db267c023a0d5cc6': 12, 'ba52555718a753637bb619200365d871': 13,
                                      '9ff508f853f8d0c755601490ef0a89c1': 14, '7a239f3e0504501beec4142fdc924938': 15,
                                      '58d86475fa4f0f2efdfbdb677e002b59': 16, 'dcd692b6c2440d8d23dde109b9e62fc5': 17,
                                      '9e871dfff4829e7757932831854619fc': 18, 'e143029ad79a8ca9bf3e1458bc91c4f3': 19,
                                      'b8332f5b648873bbc03f40e68a679161': 20, 'b95f9d3e8836cb4a8f53525cc557f8eb': 21},
                'Operating_Area_Name': {'05074bd887151b4dccb470dd3d26faad': 0, '86114fa7492257e065abca01e4753bba': 1,
                                        '00f9e34b8bdaaf007602def09837636b': 2, '6bd116d685f1a5f6b4b773faf4212114': 3},
}
