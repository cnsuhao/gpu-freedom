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
    id          : Longint;
    url,
    chatchannel : String;
end;

type TServerManager = class(TObject)
   public
    constructor Create(var conf : TCoreConfiguration; servertable : TDbServerTable;
                       var logger : TLogger);
    destructor Destroy;

    procedure getServer(var srv : TServerRecord);
    procedure getDefaultServer(var srv : TServerRecord);
    procedure getServerIndex(var srv : TServerRecord; idx : Longint);

    procedure reloadServers();
    procedure increaseFailures(url : String);

   private
    defaultserver_  : Longint;
    cs_             : TCriticalSection;
    count_,
    currentServers_ : Longint;
    servertable_    : TDbServerTable;
    logger_         : TLogger;
    conf_           : TCoreConfiguration;

    servers_        : array [1..MAX_SERVERS] of TServerRecord;

    procedure getServerInternal(var srv : TServerRecord; i : Longint);
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


procedure TServerManager.getServerIndex(var srv : TServerRecord; idx : Longint);
var i : Longint;
begin
  if (idx=-1) then
              begin
                // this is a local job
                srv.id:=-1;
                srv.url:='';
                srv.chatchannel:='';
                logger_.log(LVL_SEVERE, 'srvid:=-1 in servMan.getServerIndex!!');
                Exit;
              end;
  if (idx<2) or (idx>currentservers_) then raise Exception.Create('getServerIndex: '+IntToStr(idx)+' is out of range!!');
  // locate the server with this idx
  CS_.Enter;

  for i:=1 to currentservers_ do
    begin
      if (idx=servers_[i].id) then
             begin
                  srv.id  := servers_[i].id;
                  srv.url := servers_[i].url;
                  srv.chatchannel := servers_[i].chatchannel;
                  CS_.Leave;
                  Exit;
             end;
    end;
  CS_.Leave;
  raise Exception.Create('getServerIndex: '+IntToStr(idx)+' is not loaded in memory (and database)!!');
end;

procedure TServerManager.getServerInternal(var srv : TServerRecord; i : Longint);
begin
  srv.id  := servers_[i].id;
  srv.url := servers_[i].url;
  srv.chatchannel := servers_[i].chatchannel;
end;

procedure TServerManager.getServer(var srv : TServerRecord);
begin
  cs_.Enter;
  getServerInternal(srv, count_);
  Inc(count_);
  if count_>currentServers_ then count_ := 1;
  cs_.Leave;
end;


procedure TServerManager.getDefaultServer(var srv : TServerRecord);
begin
  cs_.Enter;
  getServerInternal(srv, defaultserver_);
  cs_.Leave;
end;


procedure TServerManager.reloadServers();
var i  : Longint;
    ds : TSqlite3DataSet;
begin
 logger_.log(LVL_DEBUG, 'TServermanager> Entering reloadServers()');
 cs_.Enter;
 i:=0;
 defaultserver_ := -1;

try
 ds := servertable_.getDS();
 ds.First;
 while not ds.EOF do
    begin
     if not ds.FieldValues['online'] then continue;
     Inc(i);
     servers_[i].url         := ds.FieldValues['serverurl'];
     servers_[i].id          := ds.FieldValues['id'];
     servers_[i].chatchannel := ds.FieldValues['chatchannel'];
     logger_.log(LVL_DEBUG, 'TServermanager> '+ds.FieldValues['serverurl']);

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
        servers_[1].id := 0;
        servers_[1].chatchannel := 'Altos';
        currentServers_ := 1;
        defaultserver_ := 1;
        logger_.log(LVL_INFO, 'TServermanager> Default superserver initially set to '+myConfID.default_superserver_url);
      end
     else
       begin
         currentServers_ := i;
         logger_.log(LVL_INFO, 'TServermanager> Default superserver now set to '+myConfID.default_superserver_url);
       end;

 count_ := defaultserver_;
except
  on E : Exception do
     begin
      logger_.log(LVL_DEBUG, 'Exception '+e.ClassName+' thrown with message '+e.Message);
     end;
end;

cs_.Leave;
logger_.log(LVL_DEBUG, 'TServermanager> Exiting reloadServers()');

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
