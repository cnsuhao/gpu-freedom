unit receiveclientservices;
{

  This unit receives a list of active XML nodes from GPU II servers
   and stores it in the TDbClientTable object.receivenodeservices

  (c) 2010 by HB9TVM and the GPU Team
  This code is licensed under the GPL

}
interface

uses coreservices, servermanagers,
     clienttables, loggers, Classes, SysUtils, DOM;

type TReceiveClientServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; proxy, port : String;
                     clienttable : TDbClientTable; var logger : TLogger);
 protected
    procedure Execute; override;

 private
   clienttable_ : TDbClientTable;

   procedure parseXml(var xmldoc : TXMLDocument);
end;

implementation

constructor TReceiveClientServiceThread.Create(var servMan : TServerManager; proxy, port : String;
                                               clienttable : TDbClientTable; var logger : TLogger);
begin
 inherited Create(servMan, proxy, port, logger);
 clienttable_ := clienttable;
end;


procedure TReceiveClientServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    dbnode   : TDbClientRow;
    node     : TDOMNode;
    port     : String;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  node := xmldoc.DocumentElement.FirstChild;

  while Assigned(node) do
    begin
        try
             begin
               dbnode.nodeid            := node.FindNode('nodeid').TextContent;
               dbnode.defaultservername := node.FindNode('defaultservername').TextContent;
               dbnode.nodename          := node.FindNode('nodename').TextContent;
               dbnode.country           := node.FindNode('country').TextContent;
               dbnode.region            := node.FindNode('region').TextContent;
               dbnode.city              := node.FindNode('city').TextContent;
               dbnode.zip               := node.FindNode('zip').TextContent;
               dbnode.description       := node.FindNode('description').TextContent;
               dbnode.ip                := node.FindNode('ip').TextContent;
               dbnode.port              := node.FindNode('port').TextContent;
               dbnode.localip           := node.FindNode('localip').TextContent;
               dbnode.os                := node.FindNode('os').TextContent;
               dbnode.cputype           := node.FindNode('cputype').TextContent;
               dbnode.version           := node.FindNode('version').TextContent;
               dbnode.acceptincoming    := (node.FindNode('acceptincoming').TextContent='true');
               dbnode.gigaflops    := StrToInt(node.FindNode('gigaflops').TextContent);
               dbnode.ram          := StrToInt(node.FindNode('ram').TextContent);
               dbnode.mhz          := StrToInt(node.FindNode('mhz').TextContent);
               dbnode.nbcpus       := StrToInt(node.FindNode('nbcpus').TextContent);
               dbnode.bits         := StrToInt(node.FindNode('bits').TextContent);
               dbnode.online  := true;
               dbnode.updated := true;
               dbnode.uptime      := StrToFloatDef(node.FindNode('uptime').TextContent, 0);
               dbnode.totaluptime := StrToFloatDef(node.FindNode('totaluptime').TextContent, 0);
               dbnode.longitude   := StrToFloatDef(node.FindNode('longitude').TextContent, 0);
               dbnode.latitude    := StrToFloatDef(node.FindNode('latitude').TextContent, 0);
               dbnode.userid      := node.FindNode('userid').TextContent;
               dbnode.team        := node.FindNode('team').TextContent;
               clienttable_.insertOrUpdate(dbnode);
               logger_.log(LVL_DEBUG, 'Updated or added <'+dbnode.nodename+'> to tbclient table.');
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveClientServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

       node := node.NextSibling;
     end;  // while Assigned(node)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveClientServiceThread.Execute;
var xmldoc    : TXMLDocument;
    url       : String;
begin
 url := servMan_.getServerUrl();
 receive(url, '/list_clients_online_xml.php',
         '[TReceiveClientServiceThread]> ', xmldoc, true);

 if not erroneous_ then
    begin
     clienttable_.execSQL('UPDATE tbnode set updated=0;');
     parseXml(xmldoc);
     if not erroneous_ then
        clienttable_.execSQL('UPDATE tbnode set online=updated;');
    end;

 finishReceive(url, '[TReceiveClientServiceThread]> ', 'Service updated table TBNODE succesfully :-)', xmldoc);
end;

end.
