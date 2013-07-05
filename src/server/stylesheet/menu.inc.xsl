<?xml version='1.0'?>
<xsl:stylesheet version="1.0"
      xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
      xml:space="preserve">
<xsl:template name="MENU">
<td>
<b>Navigation</b><br />
<br/>
<a href='index.php'>Home</a><br/>
<br/>

<b>Jobs</b><br/>
<a href='../jobqueue/list_jobdefinitions.php'>Definitions</a><br/>
<a href='../jobqueue/list_jobqueues.php'>Queue</a><br/>
<a href='../jobqueue/list_jobresults.php'>Results</a><br/>
<a href='../jobqueue/list_jobstats.php'>Stats</a><br/>
<br/>
<b>Channels</b><br/>
<a href='../channel/list_channels.php'>Channels</a><br/>
<a href='../channel/list_channel_messages.php'>Messages</a><br/>
<br/>
<b>Cluster</b><br/>
<a href='../cluster/list_clients.php'>Clients</a><br/>
<a href='../supercluster/list_servers.php'>Servers</a><br/>
<a href='../supercluster/list_parameters.php'>Parameters</a><br/>
</td>
</xsl:template>
</xsl:stylesheet>