== Description ==

OctaveGUI is a(nother) GUI frontend for GNU Octave written fully in FreePascal
with Lazarus IDE, with the following feature goals:
- Cross platform (CPU, OS, and widgetset)
- Portable
- Small size
- Fast execution & low memory consumption
- Close interface to MATLAB

== Current features ==

- Cross platform. Tested on Win32 native & Linux gtk2. Carbon & QT theoretically
  should work without problems
- Portable. Configuration file placed together with the executable*, just bring
  them both together and set some path
- Changing, saving, and loading active directory (not yet 100% save, see bugs below)
- Restarting non-responsive Octave process (useful when there's a bug)
- Clearing output prompt
- Saving output to file
- [NEW] M files browser (synchronized to current dir), double click on filename
  to open with Octave's edit command
- [NEW] History list, double click to execute command

* IMPORTANT!
  In order to work well, under Unix you MUST NOT call it via symlinks. Otherwise
  the configuration will be searched & placed along with the symlink (don't blame me,
  it's Unix fault)

== Next to implement ==

- Sessions (so that one can continue previously stopped jobs)
- Copying output to clipboard (only text in prompt can be copied for now, I'll ask the
  component author)

== Bugs ==

- Please choose the correct Octave exe at first run. If you suddenly pick the wrong one,
  just restart it. Please ignore any error message, choose cancel when asked to terminate.
  I haven't handled it correctly.
- [FIXED] Terminating the program or restarting Octave while a process is running will make it
  save wrong active directory, it would be junk from the output process or even nothing.
  If next time you start it fails with error message like "failed to execute bla bla : 267",
  edit CurrentDir variable under [Octave] section in OctaveGUI.ini. 267 is a Windows error
  code for ERROR_DIRECTORY, meaning that the directory doesn't exist. On other OSes, this
  value might differ. Please consult your platform documentation.

== System requirements ==

- A computer that can boot quite recent Windows (I never have 98 or 2000, but it's worth
  to try), Linux, FreeBSD or Mac OS X. If your computer can boot those OSes, I bet you
  already have more than 6 MB of memory* and 2 MB of disk space**. For a nice view, I suggest
  having monitor with 1024 x 768 or above resolution

*  This is for running the program, Octave itself eats about 30 MB. This might increases as
   more elements are added, but I expect in the end (don't ask when) it'll eat at most 12 MB.

** Should increase as more elements are added, but since the very complex IDE used to build it
   is only about 9 MB (including some stupid and useless components that I install), the final
   program shouldn't be bigger than that.

== (Code) Documentation ==

The source code is the docs... seriously, I'm not very good at making it (have you ever heard
"wrong docs are worse than no docs"?). I've put comments in the code to answer some of your whys.

== License ==

GPL v2 or above

== Author ==

Mario Ray Mahardhika (leledumbo_cool@yahoo.co.id)

== Main Site ==

code.google.com/p/octave-gui
