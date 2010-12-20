unit servermanagers;
{
  This unit manages GPU II servers, it can cicly give them back via
   getServerURL, or the default server url can be retrieved as well

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL
}
interface

uses SyncObjs, Sysutils, Classes, servertables, sqlite3ds, loggers,
     coreconfigurations, identities;

const MAX_SERVERS = 300;

type TServerRecord = record
    id  : Longint;
    url : String;
end;

type TServerManager = class(TObject)
   public
    constructor Create(var conf : TCoreConfiguration; servertable : TDbServerTable;
                       var logger : TLogger);
    destructor Destroy;

    procedure getServerUrl(var srv : TServerRecord);
    procedure getDefaultServerUrl(var srv : TServerRecord);
    procedure getSuperServerUrl(var srv : TServerRecord);

    procedure reloadServers();
    procedure increaseFailures(url : String);

   private
    defaultserver_  : Longint;
    superserver_    : Longint;
    cs_             : TCriticalSection;
    count_,
    currentServers_ : Longint;
    servertable_    : TDbServerTable;
    logger_         : TLogger;
    conf_           : TCoreConfiguration;

    servers_        : array [1..MAX_SERVERS] of TServerRecord;

    procedure getServer(var srv : TServerRecord; i : Longint);
end;

implementation

constructor TServerManager.Create(var conf : TCoreConfiguration; servertable : TDbServerTable;
                                  var logger : TLogger);
begin
  inherited Create;
  cs_ := TCriticalSection.Create();
  servertable_ := servertable;
  logger_ := logger;
  conf_ := conf;
  reloadServers();
end;

destructor TServerManager.Destroy;
begin
  cs_.Free;
end;

procedure TServerManager.getServer(var srv : TServerRecord; i : Longint);
begin
  srv.url := servers_[i].url;
  srv.id  := servers_[i].id
end;

procedure TServerManager.getServerUrl(var srv : TServerRecord);
begin
  cs_.Enter;
  getServer(srv, count_);
  Inc(count_);
  if count_>currentServers_ then count_ := 1;
  cs_.Leave;
end;


procedure TServerManager.getDefaultServerUrl(var srv : TServerRecord);
begin
  cs_.Enter;
  getServer(srv, defaultserver_);
  cs_.Leave;
end;

procedure TServerManager.getSuperServerUrl(var srv : TServerRecord);
begin
  cs_.Enter;
  getServer(srv, superserver_);
  cs_.Leave;
end;


procedure TServerManager.reloadServers();
var i  : Longint;
    ds : TSqlite3DataSet;
begin
 cs_.Enter;
 i:=0;
 defaultserver_ := -1;
 superserver_ := -1;

 ds := servertable_.getDS();
 ds.First;
 while not ds.EOF do
    begin
     if not ds.FieldValues['online'] then continue;
     Inc(i);
     servers_[i].url := ds.FieldValues['serverurl'];
     servers_[i].id  := ds.FieldValues['id'];
     logger_.log(LVL_DEBUG, 'TServermanager> '+ds.FieldValues['serverurl']);
     if ds.FieldValues['superserver']=true then
        begin
          superserver_ := i;
          logger_.log(LVL_DEBUG, 'TServermanager> ^ is superserver');
        end;
     if ds.FieldValues['defaultsrv']=true then
         begin
          defaultserver_ := i;
          myConfID.default_server_name:=ds.FieldValues['servername'];
          logger_.log(LVL_DEBUG, 'TServermanager> ^ is defaultserver');
         end;
     ds.Next;
    end;

 if (i=0) then
      begin
        servers_[1].url := myConfID.default_superserver_url;
        servers_[1].id := 1;
        currentServers_ := 1;
        logger_.log(LVL_INFO, 'TServermanager> Superserver initially set to '+myConfID.default_superserver_url);
      end
     else
       currentServers_ := i;

 if superserver_=-1 then
    begin
      logger_.log(LVL_SEVERE, 'TServermanager> Superserver not defined');
      superserver_ := 1;
    end;

 if defaultserver_=-1 then
    begin
      logger_.log(LVL_SEVERE, 'TServermanager> Defaultserver not defined');
      defaultserver_ := 1;
    end;

 count_ := defaultserver_;
 cs_.Leave;
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



end.
