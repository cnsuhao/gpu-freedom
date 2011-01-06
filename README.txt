GPU, a Global Processing Unit
(c) 2002-2011 the GPU Development Team

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
plugins and frontends in Freepascal/C/C++/Java and executables targeting the platform on which GPU II runs

Targeted platform
-----------------
client GUI will run on Windows, Linux and MacOSX (everywhere where Lazarus runs)
client core will run on Windows, Linux and MacOSX (everywhere where freepascal runs),
 it will also feature an interative console similar to Mutella
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
   /logs        logfiles for core
   /languages   localization directory
      /mainapp    language for main application
      /extensions language for extensions	  
   /plugins     plugins come here, managed by core
      /lib      plugin as compiled libraries
      /exe      external executables
      /jar      Java archives
   /temp        temporary directory for general use
   /workunits   directory for workunits
      /incoming incoming workunits
      /staged   workunits being processed
      /outgoing workunits which need upload to server
   /data    data for general use
      /core        data for core, including core.db which is sqllite3 database
	               and is used for communication between mainapp and core
      /extensions  data for plugins and frontends
      /planet      data on earth planet on elevation and population
      /textures    textures for general OpenGL use
        /solarysystem textures of solarsystem
   /linux-x86-32   libraries for linux support on x86, 32 bit
   /linux-x86-64   libraries for linux support on x86, 64 bit
   /macosx         libraries for macosx support
   /win32          libraries for windows support, 32 bit
   /win64          libraries for windows support, 64 bit

/dcu  compiled units for the client

/docs documentation for gpu_freedom
  /dev   documentation for developers
    /fpcunit documentation regarding fpcunit framework for unit testing (port of JUnit in FPC)
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
       /basic      plugin containg basic routines for floats
	   /strbasic   plugin containg simple routines for strings
 
     /lib          libraries for GPU
       /own        own libraries developed by the project
         /simclimate   logic of simclimate extension
         /coremodules  Core modules used by core component under client/core
		 /coreservices Core services used by core component to download/transmit data to servers
         /misc         units for general purpose (like logger)
         /coredb       sqlite database management units
       /ext        external libraries
         /synapse  ararat synapse library for Internet connections by Lukas Gebauer

     /packages     components for GPU
       /own        own components developed by the project
       /ext        external packages

/server  Server source code
   /admin    server administration part
   /channel  source code implementing channels (chat is also a channel)
   /cluster  source code for cluster management of clients
   /conf     server configuration
   /db       sql  scripts to create the database schema
   /jobqueue  source code managing the job queue
   /wustorage where workunits are stored
      /delivery    workunits ready for delivery
      /received    computed workunits received from clients
   /supercluster   source code to inform parent server, so that
                   servers can be organized hierarchically in a tree
	   /user       user management at supercluster level
   /utils      php scripts for general use
   /workunits  source code to manage workunits
 
    

Have fun!
