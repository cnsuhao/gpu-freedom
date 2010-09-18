(c) 2002-2010 the GPU Development Team

Most source is under GNU General Public License (see GPL_license.txt).
The PPL is a modified GPL license to prohibit military use (see PPL_license.txt). 
However, this code is currently licensed only under GPL.
 
Some source from external libraries and packages is under other licenses,
a note is placed either on the source code or in a license file in the respective
folder.

GPU Homepage:     http://gpu.sourceforge.net
GPU Mailing List: gpu-world@lists.sourceforge.net
GPU Team Members: http://sourceforge.net/project/memberlist.php?group_id=58134

Programming languages
---------------------
core, mainapp and screensaver in Freepascal/Lazarus
server in PHP&mySQL
plugins and frontends in Freepascal/C/C++/Java 


File Tree structure for this package, codenamed freedom follows:
----------------------------------------------------------------
/bin         binaries for the core, frontends, mainapp and screensaver
   /conf        configuration files for the client
      /core        configuration files for core, mainapp and screensaver
      /extensions   configuration files for frontends, plugins
   /jobs        jobs for the core
      /input    input jobs come here
      /loading  jobs being processed by core
      /output   result of jobs comes here
   /extensions  data for plugins and frontends
   /logs        logfiles for core
   /languages   languages for core and extensions
   /plugins     dll of plugins come here, loaded by core
   /temp        temporary directory for general use
   /workunits   directory for workunits
      /incoming incoming workunits
      /staged   workunits being processed
      /outgoing workunits which need upload to server
   /data    data for general use
      /planet      data on earth planet on elevation and population
      /textures    textures for general OpenGL use
        /solarysystem textures of solarsystem


Have fun!
