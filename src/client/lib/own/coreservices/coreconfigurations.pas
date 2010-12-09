unit coreconfigurations;

interface

uses identities, syncObjs, inifiles;

type TCoreConfiguration = class(TObject)
 public
  constructor Create(path, filename : String);
  destructor Destroy;

  function getGPUIdentity  : TGPUIdentity;
  function getUserIdentity : TUserIdentity;
  function getConfIdentity : TConfIdentity;

  procedure loadConfiguration();
  procedure saveConfiguration();

 private
  filename_,
  path_      : String;

  ini_       : TInifile;

  CS_ : TCriticalSection;
end;


implementation

constructor TCoreConfiguration.Create(path, filename : String);
begin
  inherited Create;

  path_ := path;
  filename_ := filename;

  ini_ := TInifile.Create(path_+filename_);
  CS_ := TCriticalSection.Create;
end;

destructor TCoreConfiguration.Destroy;
begin
 ini_.Free;
 CS_.Free;
 inherited Destroy;
end;


function TCoreConfiguration.getGPUIdentity  : TGPUIdentity;
begin
 Result := myGPUID;
end;


function TCoreConfiguration.getUserIdentity : TUserIdentity;
begin
 Result := myUserID;
end;

function TCoreConfiguration.getConfIdentity : TConfIdentity;
begin
 Result := myConfID;
end;

procedure TCoreConfiguration.loadConfiguration();
begin
  CS_.Enter;
  with myGPUID do
    begin
      NodeName  := ini_.ReadString('core','nodename','testnode');
      Team      := ini_.ReadString('core','team','team');
      Country   := ini_.ReadString('core','country','country');
      Region    := ini_.ReadString('core','region','region');
      Street    := ini_.ReadString('core','street','street');
      City      := ini_.ReadString('core','city','city');
      Zip       := ini_.ReadString('core','zip','zip');
      NodeId    := ini_.ReadString('core','nodeid','nodeid');
      IP        := ini_.ReadString('core','ip','ip');
      localIP   := ini_.ReadString('core','localip','localip');
      OS        := ini_.ReadString('core','os','test os');
      Version   := ini_.ReadString('core','version','1.0.0');
      Port      := ini_.ReadInteger('core','port',1234);
      AcceptIncoming := ini_.ReadBool('core','acceptincoming',false);
      MHz            := ini_.ReadInteger('core','mhz',1000);
      RAM            := ini_.ReadInteger('core','ram',512);
      GigaFlops      := ini_.ReadInteger('core','gigaflops',1);
      isSMP          := ini_.ReadBool('core','issmp',false);
      isHT           := ini_.ReadBool('core','isht',false);
      is64bit        := ini_.ReadBool('core','is64bit',false);
      isWineEmulator := ini_.ReadBool('core','iswineemulator',false);
      isRunningAsScreensaver := ini_.ReadBool('core','isrunningasscreensaver',false);
      nbCPUs                 := ini_.ReadInteger('core','nbcpus',1);
      Uptime                 := ini_.ReadFloat('core','uptime',0);
      TotalUptime            := ini_.ReadFloat('core','totaluptime',0);
      CPUType                := ini_.ReadString('core','cputype','AMD');

      Longitude              := ini_.ReadFloat('core','longitude',7);
      Latitude               := ini_.ReadFloat('core','latitude',45);
   end;

 with myUserID do
 begin
      userid        := ini_.ReadString('user','userid','1');
      username      := ini_.ReadString('user','username','testuser');
      password      := ini_.ReadString('user','password','');
      email         := ini_.ReadString('user','email','testuser@test.com');
      realname      := ini_.ReadString('user','realname','Paul Smith');
      homepage_url  := ini_.ReadString('user','homepage_url','http://www.gpu-grid.net');
 end;

 with myConfID do
 begin
      max_computations        := ini_.ReadInteger('configuration','max_computations',3);
      max_services            := ini_.ReadInteger('configuration','max_services',3);
      max_downloads           := ini_.ReadInteger('configuration','max_downloads',3);

      default_superserver_url := ini_.ReadString('configuration','default_superserver_url','http://www.gpu-grid.net/superserver');
 end;
 CS_.Leave;
end;

procedure TCoreConfiguration.saveConfiguration();
begin
  CS_.Enter;
  {
  with myGPUID do
    begin
      NodeName
      Team
      Country
      Region
      NodeId
      IP
      localIP
      OS
      Version
      Port
      AcceptIncoming
      MHz
      RAM
      GigaFlops
      isSMP
      isHT
      is64bit
      isWineEmulator
      isRunningAsScreensaver
      nbCPUs
      Uptime
      TotalUptime
      CPUType

      Longitude
      Latitude

      max_computations
      max_services
      max_downloads
  end;

  with myUserID do
      userid
      username
      password
      email
      realname
      homepage_url
  end;
  }
  CS_.Leave;
end;



end.
