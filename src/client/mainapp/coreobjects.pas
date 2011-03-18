unit coreobjects;

interface

uses
  lockfiles, loggers, SysUtils,
  coreconfigurations, dbtablemanagers, servermanagers;

var
   logger       : TLogger;
   conf         : TCoreConfiguration;
   tableman     : TDbTableManager;
   serverman    : TServerManager;


procedure loadCoreObjects;
procedure discardCoreObjects;

implementation

procedure loadCoreObjects;
var
   CO_path : String;
begin
  CO_path := extractFilePath(ParamStr(0));

  logger    := TLogger.Create(CO_path+PathDelim+'logs', 'guiapp.log', 'guiapp.old', LVL_DEBUG, 1024*1024);
  conf      := TCoreConfiguration.Create(CO_path, 'coreapp.ini');
  conf.loadConfiguration();
  tableman := TDbTableManager.Create(CO_path+PathDelim+'coreapp.db');
  tableman.OpenAll;
  serverman := TServerManager.Create(conf, tableman.getServerTable(), logger);
end;

procedure discardCoreObjects;
begin
  //conf_.saveConfiguration();

  serverman.Free;
  tableman.CloseAll;
  tableman.Free;
  conf.Free;
  logger.Free;
end;

end.
