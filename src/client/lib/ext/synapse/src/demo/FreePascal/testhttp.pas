{$MODE DELPHI}

Program testhttp;

uses
  httpsend, classes, Sysutils;

var
  HTTP: THTTPSend;
  l: tstringlist;
begin
  HTTP := THTTPSend.Create;
  if Trim(Paramstr(2))<>'' then
           HTTP.ProxyHost := ParamStr(2);
  if Trim(Paramstr(3))<>'' then
           HTTP.ProxyPort := ParamStr(3);


  l := TStringList.create;
  try
    if not HTTP.HTTPMethod('GET', Paramstr(1)) then
      begin
	writeln('ERROR');
        writeln(Http.Resultcode);
      end
    else
      begin
        writeln(Http.Resultcode, ' ', Http.Resultstring);
        writeln;
        writeln(Http.headers.text);
        writeln;
        l.loadfromstream(Http.Document);
        writeln(l.text);
     end;
  finally
    HTTP.Free;
    l.free;
  end;
end.

