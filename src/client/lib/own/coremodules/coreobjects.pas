unit coreobjects;
{
 Defines a set of objects in common between mainapp (GUI) and core
 (c) 2011 by HB9TVM and the GPU project
}


interface

uses
  lockfiles, loggers, SysUtils,
  coreconfigurations, dbtablemanagers, servermanagers,
  coremodules, servicefactories, servicemanagers,
  identities;

var
   logger         : TLogger;
   conf           : TCoreConfiguration;
   tableman       : TDbTableManager;
   serverman      : TServerManager;
   coremodule     : TCoreModule;
   servicefactory : TServiceFactory;
   serviceman     : TServiceThreadManager;

   // lockfiles
   lf_morefrequentupdates : TLockFile;



procedure loadCoreObjects(logFile : String);
procedure discardCoreObjects;

implementation

procedure loadCoreObjects(logFile : String);
var
   path : String;
begin
  path := extractFilePath(ParamStr(0));

  logger    := TLogger.Create(path+PathDelim+'logs', logFile+'.log', logFile+'.old', LVL_DEBUG, 1024*1024);
  conf      := TCoreConfiguration.Create(path);
  conf.loadConfiguration();
  tableman := TDbTableManager.Create(path+PathDelim+'coreapp.db');
  tableman.OpenAll;
  serverman := TServerManager.Create(conf, tableman.getServerTable(), logger);

  coremodule       := TCoreModule.Create(logger, path, 'dll');
  servicefactory   := TServiceFactory.Create(serverman, tableman, myConfId.proxy, myconfId.port, logger, conf);
  serviceman       := TServiceThreadManager.Create(tmServiceStatus.maxthreads);

  lf_morefrequentupdates := TLockFile.Create(path+PathDelim+'locks', 'morefrequentchat.lock');

end;

procedure discardCoreObjects;
begin
  serviceman.free;
  servicefactory.free;
  coremodule.free;

  serverman.Free;
  tableman.CloseAll;
  tableman.Free;
  conf.Free;
  logger.Free;

  lf_morefrequentupdates.Free;
end;

end.
