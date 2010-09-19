GPU, a Global Processing Unit
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

Targeted platform
-----------------
client will run on Windows, Linux and MacOSX and Windows (everywhere where Lazarus runs)
server will run on Windows, Linux, MacOSX, several Unix flavors (everywhere where PHP/mySQL run on Apache)

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

/dcu  compiled units for the client

/docs documentation for gpu_freedom
  /dev   documentation for developers
  /users documentation for users

/install
  /users install script for users
  /dev   install script for developers

/src
  /client source code for the client
     /core source code for core component (no GUI)
     /mainapp   source code for main application with GUI
     /screensaver source code for screensaver

     /frontends    source code for frontends
       /simclimate Climate simulation frontend

     /plugins      source code for plugins
       /basic      plugin containg basic routines like add and mul
 
     /lib          libraries for GPU
       /own        own libraries developed by the project
         /simclimate  logic of simclimate extension
       /ext        external libraries

     /packages     components for GPU
       /own        own components developed by the project
          /tgpu    TGPU component used in core to manage plugins
       /ext        external packages


/server  Server source code
   /chat  source code implementing chat
   /cluster  source code for cluster management of clients
   /db    scripts to create the database schema
   /jobqueue  source code managing the job queue
   /supercluster   source code to manage cluster of servers
   /workunits  source code to manage workunits
 
    

Have fun!
