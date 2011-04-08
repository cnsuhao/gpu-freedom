unit coreobjects;

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


procedure loadCoreObjects;
procedure discardCoreObjects;

implementation

procedure loadCoreObjects;
var
   path : String;
begin
  path := extractFilePath(ParamStr(0));

  logger    := TLogger.Create(path+PathDelim+'logs', 'guiapp.log', 'guiapp.old', LVL_DEBUG, 1024*1024);
  conf      := TCoreConfiguration.Create(path, 'coreapp.ini');
  conf.loadConfiguration();
  tableman := TDbTableManager.Create(path+PathDelim+'coreapp.db');
  tableman.OpenAll;
  serverman := TServerManager.Create(conf, tableman.getServerTable(), logger);

  coremodule       := TCoreModule.Create(logger, path, 'dll');
  servicefactory   := TServiceFactory.Create(serverman, tableman, myConfId.proxy, myconfId.port, logger, conf);
  serviceman       := TServiceThreadManager.Create(tmServiceStatus.maxthreads);
end;

procedure discardCoreObjects;
begin
  conf.saveConfiguration();

  serviceman.free;
  servicefactory.free;
  coremodule.free;

  serverman.Free;
  tableman.CloseAll;
  tableman.Free;
  conf.Free;
  logger.Free;
end;

end.
