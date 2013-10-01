-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Oct 01, 2013 at 09:13 AM
-- Server version: 5.5.25a
-- PHP Version: 5.4.4

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `finance`
--

-- --------------------------------------------------------

--
-- Table structure for table `tickernames`
--

CREATE TABLE IF NOT EXISTS `tickernames` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(8) NOT NULL,
  `description` varchar(255) NOT NULL,
  `type` varchar(16) DEFAULT NULL,
  `url` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=20 ;

--
-- Dumping data for table `tickernames`
--

INSERT INTO `tickernames` (`id`, `name`, `description`, `type`, `url`) VALUES
(1, 'VIX', 'VOLATILITY S&P 500', 'YAHOO', 'http://finance.yahoo.com/q?s=^VIX'),
(2, 'GSPC', 'Standard&Poors 500', 'YAHOO', 'http://finance.yahoo.com/q?s=^gspc'),
(3, 'QQQX', 'NASDAQ Premium Income and Growt', 'YAHOO', 'http://finance.yahoo.com/q?s=QQQX'),
(4, 'GOOG', 'Google', 'YAHOO', 'http://finance.yahoo.com/q?s=GOOG'),
(5, '2498.TW', 'HTC', 'YAHOO', 'http://finance.yahoo.com/q?s=HTC'),
(7, 'REPI.SW', 'Repower', 'YAHOO', 'http://finance.yahoo.com/q?s=REPI.SW'),
(8, 'AMZN', 'Amazon', 'YAHOO', 'http://finance.yahoo.com/q?s=AMZN'),
(9, 'NVDA', 'NVidia', 'YAHOO', 'http://finance.yahoo.com/q?s=NVDA'),
(10, 'SAP', 'SAP', 'YAHOO', 'http://finance.yahoo.com/q?s=NVDA'),
(11, 'SAP', 'SAP', 'YAHOO', 'http://finance.yahoo.com/q?s=SAP'),
(12, 'PLWTF', 'Panalpina', 'YAHOO', 'http://finance.yahoo.com/q?s=PLWTF'),
(13, 'INTC', 'Intel', 'YAHOO', 'http://finance.yahoo.com/q?s=INTC'),
(14, 'AMD', 'AMD', 'YAHOO', 'http://finance.yahoo.com/q?s=AMD'),
(15, 'TSLA', 'Tesla Motors', 'YAHOO', 'http://finance.yahoo.com/q?s=TSLA'),
(16, 'MSFT', 'Microsoft', 'YAHOO', 'http://finance.yahoo.com/q?s=MSFT'),
(17, 'RHT', 'Red Hat', 'YAHOO', 'http://finance.yahoo.com/q?s=RHT'),
(18, 'GKNT', 'Geeknet', 'YAHOO', 'http://finance.yahoo.com/q?s=GKNT'),
(19, 'LNKD', 'Linkedin', 'YAHOO', 'http://finance.yahoo.com/q?s=LNKD');

-- --------------------------------------------------------

--
-- Table structure for table `tickers`
--

CREATE TABLE IF NOT EXISTS `tickers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_dt` datetime NOT NULL,
  `name` varchar(8) NOT NULL,
  `value` double NOT NULL,
  `changepct` double DEFAULT NULL,
  `create_user` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=114 ;

--
-- Dumping data for table `tickers`
--

INSERT INTO `tickers` (`id`, `create_dt`, `name`, `value`, `changepct`, `create_user`) VALUES
(1, '2013-07-09 15:58:59', 'VIX', 14.78, 0, 'php'),
(2, '2013-07-09 15:59:49', 'VIX', 14.52, 0, 'php'),
(3, '2013-07-09 16:08:17', 'VIX', 14.37, 0, 'php'),
(4, '2013-07-09 16:08:21', 'VIX', 14.37, 0, 'php'),
(5, '2013-07-09 16:09:01', 'GSPC', 1645.09, 0, 'php'),
(6, '2013-07-09 16:29:09', 'GSPC', 1645.48, 0, 'php'),
(7, '2013-07-09 16:55:58', 'VIX', 14.51, 0, 'php'),
(8, '2013-07-09 16:58:16', 'GSPC', 1648.22, 0, 'php'),
(9, '2013-07-09 17:24:42', 'VIX', 14.31, 0, 'php'),
(10, '2013-07-10 08:28:56', 'VIX', 14.35, 0, 'php'),
(11, '2013-07-10 08:31:29', 'GSPC', 1652.32, 0, 'php'),
(12, '2013-07-11 09:30:00', 'VIX', 14.21, 0, 'php'),
(13, '2013-07-11 09:33:17', 'GSPC', 1652.62, 0, 'php'),
(14, '2013-07-12 07:37:12', 'VIX', 14.01, 0, 'php'),
(15, '2013-07-12 07:40:18', 'GSPC', 1675.02, 0, 'php'),
(16, '2013-07-29 09:17:20', 'VIX', 12.72, 0, 'php'),
(17, '2013-07-29 09:20:23', 'GSPC', 1691.65, 0, 'php'),
(18, '2013-07-30 09:14:07', 'VIX', 13.39, 0, 'php'),
(19, '2013-07-30 09:16:38', 'GSPC', 1685.33, 0, 'php'),
(20, '2013-07-31 08:35:32', 'VIX', 13.39, 0, 'php'),
(21, '2013-07-31 08:37:55', 'GSPC', 1685.96, 0, 'php'),
(22, '2013-08-02 09:32:01', 'VIX', 12.94, 0, 'php'),
(23, '2013-08-02 09:32:22', 'VIX', 12.94, 0, 'php'),
(24, '2013-08-02 09:34:36', 'GSPC', 1706.87, 0, 'php'),
(25, '2013-08-02 09:35:01', 'GSPC', 1706.87, 0, 'php'),
(26, '2013-08-05 08:09:30', 'VIX', 11.98, 0, 'php'),
(27, '2013-08-05 08:12:12', 'GSPC', 1709.67, 0, 'php'),
(28, '2013-08-06 08:18:36', 'VIX', 11.84, 0, 'php'),
(29, '2013-08-06 08:21:19', 'GSPC', 1707.14, 0, 'php'),
(30, '2013-08-07 08:36:31', 'VIX', 12.72, 0, 'php'),
(31, '2013-08-07 08:39:50', 'GSPC', 1697.37, 0, 'php'),
(32, '2013-08-07 17:06:48', 'VIX', 13.76, 0, 'php'),
(33, '2013-08-07 17:09:04', 'GSPC', 1687.76, 0, 'php'),
(34, '2013-08-08 13:30:39', 'VIX', 12.98, 0, 'php'),
(35, '2013-08-08 13:33:19', 'GSPC', 1690.91, 0, 'php'),
(36, '2013-08-09 07:19:20', 'VIX', 12.73, 0, 'php'),
(37, '2013-08-09 07:22:29', 'GSPC', 1697.48, 0, 'php'),
(38, '2013-08-09 09:01:06', 'VIX', 12.73, 0, 'php'),
(39, '2013-08-09 09:03:59', 'GSPC', 1697.48, 0, 'php'),
(40, '2013-08-12 08:26:00', 'VIX', 13.41, 0, 'php'),
(41, '2013-08-12 08:29:01', 'GSPC', 1691.42, 0, 'php'),
(42, '2013-08-13 08:26:02', 'VIX', 12.81, 0, 'php'),
(43, '2013-08-13 08:29:07', 'GSPC', 1689.47, 0, 'php'),
(44, '2013-08-14 08:22:13', 'VIX', 12.31, 0, 'php'),
(45, '2013-08-14 08:25:31', 'GSPC', 1694.16, 0, 'php'),
(46, '2013-08-15 07:14:23', 'VIX', 13.04, 0, 'php'),
(47, '2013-08-15 07:17:34', 'GSPC', 1685.39, 0, 'php'),
(48, '2013-08-16 14:08:01', 'VIX', 14.73, 0, 'php'),
(49, '2013-08-16 14:11:02', 'GSPC', 1661.32, 0, 'php'),
(50, '2013-08-19 11:14:04', 'VIX', 14.37, 0, 'php'),
(51, '2013-08-19 11:17:13', 'GSPC', 1655.83, 0, 'php'),
(52, '2013-08-20 08:23:45', 'VIX', 15.1, 0, 'php'),
(53, '2013-08-20 08:27:01', 'GSPC', 1646.06, 0, 'php'),
(54, '2013-08-21 09:04:32', 'VIX', 14.91, 0, 'php'),
(55, '2013-08-21 09:06:59', 'GSPC', 1652.35, 0, 'php'),
(56, '2013-08-22 08:17:07', 'VIX', 15.94, 0, 'php'),
(57, '2013-08-22 08:20:14', 'GSPC', 1642.8, 0, 'php'),
(58, '2013-08-23 07:52:59', 'VIX', 14.76, 0, 'php'),
(59, '2013-08-23 07:55:12', 'GSPC', 1656.96, 0, 'php'),
(60, '2013-08-26 08:54:14', 'VIX', 13.98, 0, 'php'),
(61, '2013-08-26 08:56:32', 'GSPC', 1663.5, 0, 'php'),
(62, '2013-08-27 08:26:12', 'VIX', 14.99, 0, 'php'),
(63, '2013-08-27 08:29:18', 'GSPC', 1656.78, 0, 'php'),
(64, '2013-08-28 08:26:15', 'VIX', 16.77, 0, 'php'),
(65, '2013-08-28 08:28:40', 'GSPC', 1630.48, 0, 'php'),
(66, '2013-08-29 09:32:04', 'VIX', 16.49, 0, 'php'),
(67, '2013-08-29 09:35:03', 'GSPC', 1634.96, 0, 'php'),
(68, '2013-08-30 07:29:09', 'VIX', 16.81, 0, 'php'),
(69, '2013-08-30 07:32:02', 'GSPC', 1638.17, 0, 'php'),
(70, '2013-09-02 08:34:37', 'VIX', 17.01, 0, 'php'),
(71, '2013-09-02 08:37:22', 'GSPC', 1632.97, 0, 'php'),
(72, '2013-09-03 09:23:14', 'VIX', 17.01, 0, 'php'),
(73, '2013-09-03 09:25:55', 'GSPC', 1632.97, 0, 'php'),
(74, '2013-09-04 08:25:37', 'VIX', 16.61, 0, 'php'),
(75, '2013-09-04 08:27:56', 'GSPC', 1639.77, 0, 'php'),
(76, '2013-09-05 08:19:50', 'VIX', 15.88, 0, 'php'),
(77, '2013-09-05 08:22:29', 'GSPC', 1653.08, 0, 'php'),
(78, '2013-09-06 09:18:45', 'VIX', 15.77, 0, 'php'),
(79, '2013-09-06 09:21:41', 'GSPC', 1655.08, 0, 'php'),
(80, '2013-09-09 07:58:49', 'VIX', 15.85, 0, 'php'),
(81, '2013-09-09 08:02:08', 'GSPC', 1655.17, 0, 'php'),
(82, '2013-09-11 11:45:04', 'VIX', 14.53, 0, 'php'),
(83, '2013-09-11 11:47:52', 'GSPC', 1683.99, 0, 'php'),
(84, '2013-09-12 08:36:32', 'VIX', 13.82, 0, 'php'),
(85, '2013-09-12 08:39:23', 'GSPC', 1689.13, 0, 'php'),
(86, '2013-09-13 07:41:24', 'VIX', 14.29, 0, 'php'),
(87, '2013-09-13 07:44:29', 'GSPC', 1683.42, 0, 'php'),
(88, '2013-09-16 08:28:09', 'VIX', 14.16, 0, 'php'),
(89, '2013-09-16 08:31:13', 'GSPC', 1687.99, 0, 'php'),
(90, '2013-09-17 09:51:18', 'VIX', 14.38, 0, 'php'),
(91, '2013-09-17 09:53:56', 'GSPC', 1697.6, 0, 'php'),
(92, '2013-09-18 08:38:56', 'VIX', 14.53, 0, 'php'),
(93, '2013-09-18 08:41:39', 'GSPC', 1704.76, 0, 'php'),
(94, '2013-09-19 08:42:28', 'VIX', 13.59, 0, 'php'),
(95, '2013-09-19 08:45:16', 'GSPC', 1725.52, 0, 'php'),
(96, '2013-09-20 07:36:13', 'VIX', 13.16, 0, 'php'),
(97, '2013-09-20 07:39:25', 'GSPC', 1722.34, 0, 'php'),
(98, '2013-09-23 08:32:17', 'VIX', 13.12, 0, 'php'),
(99, '2013-09-23 08:35:33', 'GSPC', 1709.91, 0, 'php'),
(100, '2013-09-23 16:01:11', 'VIX', 14.14, 0, 'php'),
(101, '2013-09-23 16:04:32', 'GSPC', 1703.9, 0, 'php'),
(102, '2013-09-24 08:43:59', 'VIX', 14.31, 0, 'php'),
(103, '2013-09-24 08:47:18', 'GSPC', 1701.84, 0, 'php'),
(104, '2013-09-25 08:37:36', 'VIX', 14.08, 0, 'php'),
(105, '2013-09-25 08:40:54', 'GSPC', 1697.42, 0, 'php'),
(106, '2013-09-26 08:25:07', 'VIX', 14.01, 0, 'php'),
(107, '2013-09-26 08:27:31', 'GSPC', 1692.77, 0, 'php'),
(108, '2013-09-27 08:38:38', 'VIX', 14.06, 0, 'php'),
(109, '2013-09-27 08:41:10', 'GSPC', 1698.67, 0, 'php'),
(110, '2013-09-30 08:13:54', 'VIX', 15.46, 0, 'php'),
(111, '2013-09-30 08:16:58', 'GSPC', 1691.75, 0, 'php'),
(112, '2013-10-01 08:51:13', 'VIX', 16.6, 0, 'php'),
(113, '2013-10-01 08:53:38', 'GSPC', 1681.55, 0, 'php');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
