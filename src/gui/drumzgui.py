# The GUI of Drumz is build using tkinter
import tkinter as tk
import os
from tkinter import filedialog

# The GUI contains a field to enter the path to a directory, a button to find and display the name
# of the sound files in that directory, and a button to exit the configuration GUI and start a DRUMZ session
class DrumzGUI:
    def __init__(self, master=None):

        # building the GUI using tk
        self.mainwindow = tk.Tk() if master is None else tk.Toplevel(master)

        # potentialSoundFiles stores the files found in a directory
        self.potentialSoundFiles = []

        # soundFiles stores the names of the files displayed in the config GUI
        self.soundFiles = [tk.StringVar(value="First sound not found"), tk.StringVar(value="Second sound not found"),
                           tk.StringVar(value="Third sound not found"), tk.StringVar(value="Fourth sound not found"),
                           tk.StringVar(value="Fifth sound not found")]
        
        self.frame1 = tk.Frame(self.mainwindow, container='false')

        # Title of the GUI
        self.label2 = tk.Label(self.frame1)
        self.label2.configure(background='#050040', font='{Arial} 20 {bold}', foreground='#ffffff',
                              text='DRUMZ CONFIGURATION')
        self.label2.pack(pady='15', side='top')

        # Instructions to provide path to directory with audio files
        self.label4 = tk.Label(self.frame1)
        self.label4.configure(anchor='n', background='#050040', font='{Arial} 11 {italic}', foreground='#ffffff')
        self.label4.configure(relief='flat', text='Enter the full path to a directory to scan,')
        self.label4.pack(pady='15', side='top')

        # Entry field to enter a directory path
        self.directorypath = tk.Entry(self.frame1)
        self.directorypathstring = tk.StringVar(value="/")
        self.directorypath.configure(background='#ffffff', cursor='arrow', exportselection='true',
                                     font='{Arial} 10 {}', textvariable=self.directorypathstring)
        self.directorypath.configure(foreground='#000000', highlightcolor='#000000', insertbackground='#050040',
                                     width='40')
        self.directorypath.pack(side='top')

        # Button to trigger directory scan
        self.scandirectory = tk.Button(self.frame1)
        self.scandirectory.configure(activebackground='#050040', activeforeground='#ffffff', background='#8787e7',
                                     cursor='hand2')
        self.scandirectory.configure(font='{Arial} 9 {}', foreground='#ffffff', text='Scan Directory')
        self.scandirectory.pack(pady='20', side='top')
        self.scandirectory.configure(command=self.scan_directory)

        # Label
        self.label5 = tk.Label(self.frame1)
        self.label5.configure(background='#050040', font='{Arial} 11 {italic}', foreground='#ffffff',
                              text='browse individual sound files using File Explorer,')
        self.label5.pack(pady='5', side='top')
        # Label
        self.label8 = tk.Label(self.frame1)
        self.label8.configure(background='#050040', font='{Arial} 8 {italic}', foreground='#ffffff',
                              text='(PROTIP: you can browse for the first file in a directory then scan to find the remaining files)')
        self.label8.pack(pady='0', side='top')

        # Button to browse for files
        self.browse = tk.Button(self.frame1)
        self.browse.configure(activebackground='#050040', activeforeground='#ffffff', background='#8787e7',
                                     cursor='hand2')
        self.browse.configure(font='{Arial} 9 {}', foreground='#ffffff', text='Browse')
        self.browse.pack(pady='5', side='top')
        self.browse.configure(command=self.browse_files)

        # Last of the instrutions
        self.label6 = tk.Label(self.frame1)
        self.label6.configure(background='#050040', font='{Arial} 11 {italic}', foreground='#ffffff',
                              text='or just click play now to load default sounds!')
        self.label6.pack(pady='15', side='top')

        # Button to start using DRUMZ
        self.start = tk.Button(self.frame1)
        self.start.configure(activebackground='#050040', activeforeground='#ffffff', background='#8787e7',
                             cursor='hand2')
        self.start.configure(foreground='#ffffff', text='Play Now')
        self.start.configure(command=self.start_playing)
        self.start.pack(ipadx='10', pady='0', side='top')

        # Last of the instrutions
        self.label7 = tk.Label(self.frame1)
        self.label7.configure(background='#050040', font='{Arial} 12 {bold}', foreground='#ffffff',
                              text='Sound Files Selected:')
        self.label7.pack(pady='25', side='top')

        # Entry for the first sound file
        self.file1 = tk.Entry(self.frame1)

        self.file1.configure(background='#ffffff', font='{Arial} 11 {}', foreground='#000000',
                             highlightbackground='#000000', textvariable=self.soundFiles[0])
        self.file1.configure(insertbackground='#050040', state='normal', width='40')
        _text_ = '''First sound not found'''
        self.file1.delete('0', 'end')
        self.file1.insert('0', _text_)
        self.file1.pack(pady='2', side='top')

        # Entry for the second sound file
        self.file2 = tk.Entry(self.frame1)
        self.file2.configure(background='#ffffff', font='{Arial} 11 {}', foreground='#000000',
                             highlightbackground='#000000', textvariable=self.soundFiles[1])
        self.file2.configure(insertbackground='#050040', width='40')
        _text_ = '''Second sound not found'''
        self.file2.delete('0', 'end')
        self.file2.insert('0', _text_)
        self.file2.pack(pady='2', side='top')

        # Entry for the third sound file
        self.file3 = tk.Entry(self.frame1)
        self.file3.configure(background='#ffffff', font='{Arial} 11 {}', foreground='#000000',
                             highlightbackground='#000000', textvariable=self.soundFiles[2])
        self.file3.configure(insertbackground='#050040', width='40')
        _text_ = '''Third sound not found'''
        self.file3.delete('0', 'end')
        self.file3.insert('0', _text_)
        self.file3.pack(pady='2', side='top')

        # Entry for the fourth sound file
        self.file4 = tk.Entry(self.frame1)
        self.file4.configure(background='#ffffff', font='{Arial} 11 {}', foreground='#000000',
                             highlightbackground='#000000', textvariable=self.soundFiles[3])
        self.file4.configure(insertbackground='#050040', width='40')
        _text_ = '''Fourth sound not found'''
        self.file4.delete('0', 'end')
        self.file4.insert('0', _text_)
        self.file4.pack(pady='2', side='top')

        # Entry for the fifth sound file
        self.file5 = tk.Entry(self.frame1)
        self.file5.configure(background='#ffffff', font='{Arial} 11 {}', foreground='#000000',
                             highlightbackground='#000000', textvariable=self.soundFiles[4])
        self.file5.configure(insertbackground='#050040', width='40')
        _text_ = '''Joe is bad at coding'''
        self.file5.delete('0', 'end')
        self.file5.insert('0', _text_)
        self.file5.pack(pady='2', side='top')

        self.frame1.configure(background='#050040', height='600', takefocus=False, width='800')
        self.frame1.pack(side='top')
        self.mainwindow.configure(background='#050040', height='200', relief='flat', width='200')
        self.mainwindow.geometry('800x600')
        self.mainwindow.resizable(False, False)

        # Main widget
        self.mainwindow = self.mainwindow

    def run(self):
        self.mainwindow.mainloop()

    # Called when button to scan directory is clicked
    def scan_directory(self):
        directoryPath = self.directorypathstring.get()

        # if the directory path is empty we don't do anything
        if directoryPath:
            allFilesInDirectory = os.listdir(directoryPath)
            self.potentialSoundFiles.clear()

            # first need to make sure we only keep mp3 and wav files
            for i in range(len(allFilesInDirectory)):
                name, extension = os.path.splitext(allFilesInDirectory[i])

                if extension == ".mp3" or extension == ".wav":
                    self.potentialSoundFiles.append(allFilesInDirectory[i])

            # ensure that no more than 5 sound files are selected
            if len(self.potentialSoundFiles) > 5:
                self.potentialSoundFiles = self.potentialSoundFiles[:5]

            # set the text of the entries to the names of the sound files found
            for i in range(len(self.potentialSoundFiles)):
                self.soundFiles[i].set(self.potentialSoundFiles[i])

    # called when browse button is clicked
    def browse_files(self):
    	# a dialog is shown in the base directory of the system asking for an mp3 or wav file
    	filename = filedialog.askopenfilename(initialdir = self.directorypathstring,
                                          title = "Select Sounds",
                                          filetypes = (("Audio Files",
                                                        "*.mp3 *.wav*"),
                                                       ("all files",
                                                        "*.*")))

    	# we extract the basename of the file and the directory path (to remember where to browse for the next file)
    	file_basename = os.path.basename(filename)
    	self.directorypathstring.set(os.path.dirname(filename))

    	if file_basename:
    		name, extension = os.path.splitext(file_basename)
    		if extension == ".mp3" or extension == ".wav":
    			if len(self.potentialSoundFiles) == 5:
    				self.potentialSoundFiles[4] = self.potentialSoundFiles[3]
    				self.potentialSoundFiles[3] = self.potentialSoundFiles[2]
    				self.potentialSoundFiles[2] = self.potentialSoundFiles[1]
    				self.potentialSoundFiles[1] = self.potentialSoundFiles[0]
    				self.potentialSoundFiles[0] = file_basename

    			elif len(self.potentialSoundFiles) < 5:
    				self.potentialSoundFiles.append(file_basename) 

    			for i in range(len(self.potentialSoundFiles)):
    				self.soundFiles[i].set(self.potentialSoundFiles[i])

    # called when the play now button is clicked
    def start_playing(self):
        if len(self.potentialSoundFiles) > 0:
            print(self.potentialSoundFiles)


if __name__ == '__main__':
    app = DrumzGUI()
    app.run()
