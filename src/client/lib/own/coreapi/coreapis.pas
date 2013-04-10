unit coreapis;
{
     TCoreApi is the mother of all APIs which interface to the core client.
     It needs to be initialized with the TDbTableManager as the core client
     interfaces with the outside world through the sqlite database.

     (c) 2002-2013 by HB9TVM and the Global Processing Unit Team
     The source code is under GPL license.
}
interface

uses SysUtils, dbtablemanagers, servermanagers, loggers;

type TCoreAPI = class(TObject)
   public
     constructor Create(var tableman : TDbTableManager; var servman : TServerManager;
                        var logger : TLogger);
   protected
     tableman_  : TDbTableManager;
     servman_   : TServerManager;
     logger_    : TLogger;
     logHeader_ : String;
     appPath_   : String;
end;

implementation


constructor TCoreAPI.Create(var tableman : TDbTableManager; var servman : TServerManager; var logger : TLogger);
begin
 tableman_  := tableman;
 servman_   := servman;
 logger_    := logger;
 logHeader_ := 'TCoreAPI';
 appPath_   := ExtractFilePath(ParamStr(0));
end;

end.
