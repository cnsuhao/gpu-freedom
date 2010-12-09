unit receivenodeservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbNodeTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers,
     nodetables, loggers, Classes, SysUtils, DOM;

type TReceiveNodeServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String;
                     nodetable : TDbNodeTable; var logger : TLogger);
 protected
    procedure Execute; override;

 private
   nodetable_ : TDbNodeTable;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveNodeServiceThread.Create(var servMan : TServerManager; proxy, port : String;
                                             nodetable : TDbNodeTable; var logger : TLogger);
begin
 inherited Create(servMan, proxy, port, logger);
 nodetable_ := nodetable;
end;


procedure TReceiveNodeServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbnode   : TDbNodeRow;
    node     : TDOMNode;
    port     : String;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbnode.nodeid :=node.FindNode('nodeid').TextContent;
               //TODO: dbnode.defaultserver_id
               dbnode.nodename :=node.FindNode('processor').TextContent;
               dbnode.country :=node.FindNode('country').TextContent;
               dbnode.region :=node.FindNode('region').TextContent;
               dbnode.ip :=node.FindNode('ip').TextContent;
               port := node.FindNode('port').TextContent;
               if port='' then port:='0';
               dbnode.port :=StrToInt(port);
               //TODO: EAccessViolation dbnode.localip :=node.FindNode('localip').TextContent;
               dbnode.os :=node.FindNode('operatingsystem').TextContent;
               dbnode.cputype :=node.FindNode('cputype').TextContent;
               dbnode.version :=node.FindNode('version').TextContent;
               dbnode.acceptincoming :=(node.FindNode('accept').TextContent='1');
               dbnode.gigaflops :=StrToInt(node.FindNode('speed').TextContent);
               dbnode.ram :=StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz :=StrToInt(node.FindNode('mhz').TextContent);
               //dbnode.nbcpus :=StrToInt(node.FindNode('cpus').TextContent); empty
               dbnode.online := true;
               dbnode.updated := true;
               dbnode.uptime :=StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime :=StrToFloatDef(node.FindNode('totuptime').TextContent, 0);
               dbnode.longitude :=StrToFloatDef(node.FindNode('geolocation_x').TextContent, 0);
               dbnode.latitude :=StrToFloatDef(node.FindNode('geolocation_y').TextContent, 0);
               nodetable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Updated or added <'+dbnode.nodename+'> to tbnode table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveNodeServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveNodeServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive(servMan_.getServerUrl()+'/list_computers_online_xml.php',
         '[TReceiveNodeServiceThread]> ', xmldoc, true);

 if not erroneous_ then
    begin
     nodetable_.execSQL('UPDATE tbnode set updated=0;');
     parseXml(xmldoc);
     if not erroneous_ then
        nodetable_.execSQL('UPDATE tbnode set online=updated;');
    end;

 finish('[TReceiveNodeServiceThread]> ', 'Service updated table TBNODE succesfully :-)', xmldoc);
end;

end.
