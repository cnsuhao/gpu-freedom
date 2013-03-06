unit receiveparamservices;

interface

uses coreservices, servermanagers, dbtablemanagers,
     loggers, coreconfigurations,
     Classes, SysUtils, DOM, identities;

type TReceiveParamServiceThread = class(TReceiveServiceThread)
 public
  constructor Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                     var conf : TCoreConfiguration; var tableman : TDbTableManager);

 protected
   procedure Execute; override;

 private
   procedure parseXml(var xmldoc : TXMLDocument);

end;



implementation

constructor TReceiveParamServiceThread.Create(var servMan : TServerManager; var srv : TServerRecord; proxy, port : String; var logger : TLogger;
                                              var conf : TCoreConfiguration; var tableman : TDbTableManager);
begin
 inherited Create(servMan, srv, proxy, port, logger, '[TReceiveParamServiceThread]> ', conf, tableman);
end;

procedure TReceiveParamServiceThread.parseXml(var xmldoc : TXMLDocument);
var
    param     : TDOMNode;
begin
  logger_.log(LVL_DEBUG, 'Parsing of XML started...');
  param := xmldoc.DocumentElement.FirstChild;

  if Assigned(param) then
    begin
        try
             begin
              with myConfId do
               begin

                 {
                 receive_servers_each  := StrToInt(param.FindNode('receive_servers_each').TextContent);
                 receive_nodes_each    := StrToInt(param.FindNode('receive_nodes_each').TextContent);
                 transmit_node_each    := StrToInt(param.FindNode('transmit_node_each').TextContent);
                 receive_jobs_each     := StrToInt(param.FindNode('receive_jobs_each').TextContent);
                 transmit_jobs_each    := StrToInt(param.FindNode('transmit_jobs_each').TextContent);
                 receive_channels_each := StrToInt(param.FindNode('receive_channels_each').TextContent);
                 transmit_channels_each:= StrToInt(param.FindNode('transmit_channels_each').TextContent);
                 receive_chat_each     := StrToInt(param.FindNode('receive_chat_each').TextContent);
                 purge_server_after_failures := StrToInt(param.FindNode('purge_server_after_failures').TextContent);
                 }
                 logger_.log(LVL_DEBUG, 'All parameters updated succesfully');
               end; // with
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, logHeader_+'Exception catched in parseXML: '+E.Message);
              end;
          end; // except

     end;  // if Assigned(param)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveParamServiceThread.Execute;
var xmldoc    : TXMLDocument;
begin
 receive('/supercluster/list_parameters.php?xml=1', xmldoc, false);

 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive('Service retrieved global parameters from superserver succesfully :-)', xmldoc);
end;




end.
