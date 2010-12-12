unit servermanagers;
{
  This unit manages GPU II servers, it can cicly give them back via
   getServerURL, or the default server url can be retrieved as well

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL
}
interface

uses SyncObjs, Sysutils, Classes, servertables, sqlite3ds, loggers,
     coreconfigurations;

type TServerManager = class(TObject)
   public
    constructor Create(var conf : TCoreConfiguration; servertable : TDbServerTable;
                       var logger : TLogger);
    destructor Destroy;

    function getServerUrl : String;
    function getDefaultServerUrl : String;
    function getSuperServerUrl : String;

    procedure reloadServers();
    procedure increaseFailures(url : String);

   private
    defaultserver_ : Longint;
    superserver_   : Longint;
    urls_          : TStringList;
    cs_            : TCriticalSection;
    count_         : Longint;
    servertable_   : TDbServerTable;
    logger_        : TLogger;
    conf_          : TCoreConfiguration;

    procedure verify();
end;

implementation

constructor TServerManager.Create(var conf : TCoreConfiguration; servertable : TDbServerTable;
                                  var logger : TLogger);
begin
  inherited Create;
  urls_ := TStringList.Create;
  cs_ := TCriticalSection.Create();
  servertable_ := servertable;
  logger_ := logger;
  conf_ := conf;
  reloadServers();
  verify();
end;

destructor TServerManager.Destroy;
begin
  urls_.Free;
  cs_.Free;
end;

function TServerManager.getServerUrl : String;
begin
  cs_.Enter;
  Result := urls_.Strings[count_];
  Inc(count_);
  if (count_)>(urls_.Count-1) then count_ := 0;
  cs_.Leave;
end;

function TServerManager.getDefaultServerUrl : String;
begin
  cs_.Enter;
  Result := urls_.Strings[defaultserver_];
  cs_.Leave;
end;

function TServerManager.getSuperServerUrl : String;
begin
  cs_.Enter;
  Result := urls_.Strings[superserver_];
  cs_.Leave;
end;


procedure TServerManager.reloadServers();
var i  : Longint;
    ds : TSqlite3DataSet;
begin
 cs_.Enter;
 i:=0;
 urls_.Clear;
 defaultserver_ := -1;
 superserver_ := -1;
 count_ := 0;

 ds := servertable_.getDS();
 ds.First;
 while not ds.EOF do
    begin
     if not ds.FieldValues['online'] then continue;

     urls_.add(ds.FieldValues['serverurl']);
     logger_.log(LVL_DEBUG, 'TServermanager> '+ds.FieldValues['serverurl']);
     if ds.FieldValues['superserver']=true then
        begin
          superserver_ := i;
          logger_.log(LVL_DEBUG, 'TServermanager> ^ is superserver');
        end;
     if ds.FieldValues['defaultsrv']=true then
         begin
          defaultserver_ := i;
          conf_.getConfIdentity.default_server_name:=ds.FieldValues['servername'];
          logger_.log(LVL_DEBUG, 'TServermanager> ^ is defaultserver');
         end;
     ds.Next;
     Inc(i);
    end;

 if (i=0) then
      begin
        urls_.add(conf_.getConfIdentity.default_superserver_url);
        defaultserver_ := 0;
        superserver_ := 0;
        logger_.log(LVL_INFO, 'TServermanager> Superserver initially set to '+conf_.getConfIdentity.default_superserver_url);
      end;

 if superserver_=-1 then
    begin
      logger_.log(LVL_SEVERE, 'TServermanager> Superserver not defined');
      superserver_ := 0;
    end;

 if defaultserver_=-1 then
    begin
      logger_.log(LVL_SEVERE, 'TServermanager> Defaultserver not defined');
      defaultserver_ := 0;
    end;

 count_ := defaultserver_;
 cs_.Leave;
 verify();
end;

procedure TServerManager.increaseFailures(url : String);
begin
 cs_.Enter;
 if servertable_.increaseFailures(url) then
       logger_.log(LVL_DEBUG, 'TServermanager> Increased failure number for url: '+url)
     else
       logger_.log(LVL_SEVERE, 'TServermanager> Could not increase failure number for url: '+url);
 cs_.Leave;
end;

procedure TServerManager.verify();
begin
  if urls_.Count = 0 then
     raise Exception.Create('TServerManager: url_ is empty');
  if defaultserver_ > (urls_.Count-1) then
      raise Exception.Create('TServerManager: defaultserver is out of range (>'+IntToStr(urls_.Count-1)+')');
  if superserver_ > (urls_.Count-1) then
      raise Exception.Create('TServerManager: superserver is out of range (>'+IntToStr(urls_.Count-1)+')');
end;

end.
