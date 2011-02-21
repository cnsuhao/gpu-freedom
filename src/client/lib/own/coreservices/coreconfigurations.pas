unit coreconfigurations;

interface

uses identities, syncObjs, inifiles;

type TCoreConfiguration = class(TObject)
 public
  constructor Create(path, filename : String);
  destructor Destroy;

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
      description := ini_.ReadString('core','description','');
      NodeId    := ini_.ReadString('core','nodeid','nodeid');
      IP        := ini_.ReadString('core','ip','ip');
      localIP   := ini_.ReadString('core','localip','localip');
      OS        := ini_.ReadString('core','os','test os');
      Version   := ini_.ReadString('core','version','1.0.0');
      Port      := ini_.ReadString('core','port','1234');
      AcceptIncoming := ini_.ReadBool('core','acceptincoming',false);
      MHz            := ini_.ReadInteger('core','mhz',1000);
      RAM            := ini_.ReadInteger('core','ram',512);
      GigaFlops      := ini_.ReadInteger('core','gigaflops',1);
      bits           := ini_.ReadInteger('core','bits',32);
      isScreensaver  := ini_.ReadBool('core','isrscreensaver',false);
      nbCPUs         := ini_.ReadInteger('core','nbcpus',1);
      Uptime         := ini_.ReadFloat('core','uptime',0);
      TotalUptime    := ini_.ReadFloat('core','totaluptime',0);
      CPUType        := ini_.ReadString('core','cputype','AMD');

      Longitude      := ini_.ReadFloat('core','longitude',7);
      Latitude       := ini_.ReadFloat('core','latitude',45);
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
      max_computations        := ini_.ReadInteger('local','max_computations',3);
      max_services            := ini_.ReadInteger('local','max_services',3);
      max_downloads           := ini_.ReadInteger('local','max_downloads',3);

      run_only_when_idle      := ini_.ReadBool('local','run_only_when_idle',true);

      proxy                   := ini_.ReadString('communication','proxy','');
      port                    := ini_.ReadString('communication','port','');
      default_superserver_url := ini_.ReadString('communication','default_superserver_url','http://www.gpu-grid.net/superserver');
      default_server_name     := ini_.ReadString('communication', 'default_server_name','Altos');

      receive_servers_each   := ini_.ReadInteger('global','receive_servers_each',14400);
      receive_nodes_each     := ini_.ReadInteger('global','receive_nodes_each',120);
      transmit_node_each     := ini_.ReadInteger('global','transmit_node_each',180);
      receive_jobs_each      := ini_.ReadInteger('global','receive_jobs_each',120);
      transmit_jobs_each     := ini_.ReadInteger('global','transmit_jobs_each',120);
      receive_channels_each  := ini_.ReadInteger('global','receive_channels_each',120);
      transmit_channels_each := ini_.ReadInteger('global','transmit_channels_each',130);
      receive_chat_each      := ini_.ReadInteger('global','receive_chat_each',45);
      purge_server_after_failures := ini_.ReadInteger('global','purge_server_after_failures',30);
 end;

 with tmCompStatus do
 begin
    maxthreads := ini_.ReadInteger('computations','max_threads', 2);
 end;

 with tmDownStatus do
 begin
    maxthreads := ini_.ReadInteger('downloads','max_threads', 3);
 end;

 with tmUploadStatus do
 begin
    maxthreads := ini_.ReadInteger('uploads','max_threads', 2);
 end;

 with tmServiceStatus do
 begin
    maxthreads := ini_.ReadInteger('services','max_threads', 5);
 end;

 CS_.Leave;
end;

procedure TCoreConfiguration.saveConfiguration();
begin
  CS_.Enter;
  with myGPUID do
    begin
      ini_.WriteString('core','nodename', nodename);
      ini_.WriteString('core','team', team);
      ini_.WriteString('core','country', country);
      ini_.WriteString('core','region', region);
      ini_.WriteString('core','street', street);
      ini_.WriteString('core','city', city);
      ini_.WriteString('core','zip', zip);
      ini_.WriteString('core','description','');
      ini_.WriteString('core','nodeid', nodeid);
      ini_.WriteString('core','ip', ip);
      ini_.WriteString('core','localip', localip);
      ini_.WriteString('core','os', os);
      ini_.WriteString('core','version', version);
      ini_.WriteString('core','port', port);
      ini_.WriteBool('core','acceptincoming', acceptincoming);
      ini_.WriteInteger('core','mhz', mhz);
      ini_.WriteInteger('core','ram', ram);
      ini_.WriteInteger('core','gigaflops', gigaflops);
      ini_.WriteInteger('core','bits', bits);
      ini_.WriteBool('core','isrscreensaver', isscreensaver);
      ini_.WriteInteger('core','nbcpus', nbcpus);
      ini_.WriteFloat('core','uptime', uptime);
      ini_.WriteFloat('core','totaluptime', totaluptime);
      ini_.WriteString('core','cputype', cputype);

      ini_.WriteFloat('core','longitude', longitude);
      ini_.WriteFloat('core','latitude', latitude);
   end;

 with myUserID do
 begin
      ini_.WriteString('user','userid', userid);
      ini_.WriteString('user','username', username);
      ini_.WriteString('user','password', password);
      ini_.WriteString('user','email', email);
      ini_.WriteString('user','realname', realname);
      ini_.WriteString('user','homepage_url', homepage_url);
 end;

 with myConfID do
 begin
     ini_.WriteInteger('local','max_computations', max_computations);
     ini_.WriteInteger('local','max_services', max_services);
     ini_.WriteInteger('local','max_downloads', max_downloads);

     ini_.WriteBool('local','run_only_when_idle', run_only_when_idle);

     ini_.WriteString('communication','proxy', proxy);
     ini_.WriteString('communication','port', port);
     ini_.WriteString('communication','default_superserver_url', default_superserver_url);
     ini_.WriteString('communication', 'default_server_name', default_server_name);

     ini_.WriteInteger('global','receive_servers_each', receive_servers_each);
     ini_.WriteInteger('global','receive_nodes_each', receive_nodes_each);
     ini_.WriteInteger('global','transmit_node_each', transmit_node_each);
     ini_.WriteInteger('global','receive_jobs_each', receive_jobs_each);
     ini_.WriteInteger('global','transmit_jobs_each', transmit_jobs_each);
     ini_.WriteInteger('global','receive_channels_each', receive_channels_each);
     ini_.WriteInteger('global','transmit_channels_each', transmit_channels_each);
     ini_.WriteInteger('global','receive_chat_each', receive_chat_each);
     ini_.WriteInteger('global','purge_server_after_failures', purge_server_after_failures);
 end;

 with tmCompStatus do
 begin
    ini_.WriteInteger('computations','max_threads', maxthreads);
 end;

 with tmDownStatus do
 begin
    ini_.WriteInteger('downloads','max_threads', maxthreads);
 end;

 with tmUploadStatus do
 begin
    ini_.WriteInteger('uploads','max_threads', maxthreads);
 end;

 with tmServiceStatus do
 begin
    ini_.WriteInteger('services','max_threads', maxthreads);
 end;

 CS_.Leave;
end;



end.
