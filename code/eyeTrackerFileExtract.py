import os
import glob
import shutil
import os.path

class systemFilesOperating:
    def __init__(self,workDir):
        self.workDir = workDir

    def traversefile(self,workDir):
        newDir = "E:\\硕士课题原始数据\\脑电\\Part1"

        if not os.path.exists(newDir):
            os.makedirs(newDir)
        # shutil.copy(r'E:\硕士课题原始数据\预实验二近红外原始数据\0709_WT\H1.data',r"E:\硕士课题原始数据")
        for folder in os.listdir(workDir):
            # print(folder)
            # # 建立空路径文件夹
            newFolder = os.path.join(newDir,folder)
            print(newFolder)
            if not os.path.exists(newFolder):
                os.makedirs(newFolder)
            folderDir = os.path.join(workDir,folder)
            for txtFile in glob.glob(r"%s\autoICA_*"%folderDir):
                # print(txtFile)
                # print("%s\\" %newFolder)
                shutil.copy(txtFile,"%s\\" %newFolder)

        return 0


workDir = "G:\ica EEG\电脑1\Part2"
traverse = systemFilesOperating(workDir)
traverse.traversefile(workDir)







#
#
# import os
# import glob
# import shutil
# import os.path
#
# class systemFilesOperating:
#     def __init__(self,workDir):
#         self.workDir = workDir
#
#     def traversefile(self,workDir):
#         newDir = "E:\硕士课题原始数据\眼动数据"
#         count = 1
#         print(count)
#         # shutil.copy(r'E:\硕士课题原始数据\预实验二近红外原始数据\0709_WT\H1.data',r"E:\硕士课题原始数据")
#         for folder in os.listdir(workDir):
#             # # 建立空路径文件夹
#             newFolder = os.path.join(newDir,folder)
#             # print(newFolder)
#             os.makedirs(newFolder)
#             folderDir = os.path.join(workDir,folder)
#             for txtFile in glob.glob(r"%s\*.txt"%folderDir):
#
#                 shutil.copy(txtFile,"%s\\" %newFolder)
#
#         return count
#
#
# workDir = "G:\眼动"
# traverse = systemFilesOperating(workDir)
# traverse.traversefile(workDir)