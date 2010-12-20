unit receiveparamservices;

interface

uses coreservices, servermanagers,
     loggers, coreconfigurations,
     Classes, SysUtils, DOM, identities;

type TReceiveParamServiceThread = class(TReceiveServiceThread)
 public
   constructor Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                      var conf : TCoreConfiguration);
 protected
   procedure Execute; override;

 private
   conf_ : TCoreConfiguration;
   procedure parseXml(var xmldoc : TXMLDocument);

end;



implementation

constructor TReceiveParamServiceThread.Create(var servMan : TServerManager; proxy, port : String; var logger : TLogger;
                                              var conf : TCoreConfiguration);
begin
 inherited Create(servMan, proxy, port, logger);
 conf_ := conf;
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
                 receive_servers_each  := StrToInt(param.FindNode('receive_servers_each').TextContent);
                 receive_nodes_each    := StrToInt(param.FindNode('receive_nodes_each').TextContent);
                 transmit_node_each    := StrToInt(param.FindNode('transmit_node_each').TextContent);
                 receive_jobs_each     := StrToInt(param.FindNode('receive_jobs_each').TextContent);
                 transmit_jobs_each    := StrToInt(param.FindNode('transmit_jobs_each').TextContent);
                 receive_channels_each := StrToInt(param.FindNode('receive_channels_each').TextContent);
                 transmit_channels_each:= StrToInt(param.FindNode('transmit_channels_each').TextContent);
                 receive_chat_each     := StrToInt(param.FindNode('receive_chat_each').TextContent);
                 purge_server_after_failures := StrToInt(param.FindNode('purge_server_after_failures').TextContent);
                 logger_.log(LVL_DEBUG, 'All parameters updated succesfully');
               end; // with
             end;
          except
           on E : Exception do
              begin
                erroneous_ := true;
                logger_.log(LVL_SEVERE, '[TReceiveParamServiceThread]> Exception catched in parseXML: '+E.Message);
              end;
          end; // except

     end;  // if Assigned(param)

   logger_.log(LVL_DEBUG, 'Parsing of XML over.');
end;

procedure TReceiveParamServiceThread.Execute;
var xmldoc    : TXMLDocument;
    srv       : TServerRecord;
begin
 servMan_.getSuperServerUrl(srv);
 receive(srv, '/supercluster/get_parameters.php',
         '[TReceiveParamServiceThread]> ', xmldoc, true);

 if not erroneous_ then
     parseXml(xmldoc);

 finishReceive(srv, '[TReceiveParamServiceThread]> ', 'Service retrieved global parameters from superserver succesfully :-)', xmldoc);
end;




end.
