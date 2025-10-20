#' Compute Two-Sample Signed Kolmogorov-Smirnov Statistic
#'
#' Calculates the signed and unsigned Kolmogorov-Smirnov statistic between two samples.
#'
#' @param sample1 A numeric vector.
#' @param sample2 A numeric vector.
#' @param ecdf1 An empirical cumulative distribution function (ECDF) for sample1.
#'
#' @return A numeric vector with the KS statistic and the signed KS statistic.
#' @export

two_sample_signed_ks_statistic <- function(sample1, sample2, ecdf1 = NULL) {
  if (is.null(ecdf1)) ecdf1 <- ecdf(sample1)
  ecdf2 <- ecdf(sample2)

  all_x <- sort(unique(c(sample1, sample2)))
  y1_interp <- ecdf1(all_x)
  y2_interp <- ecdf2(all_x)
  differences <- y1_interp - y2_interp

  ks_statistic <- max(abs(differences))
  signed_ks_statistic <- differences[which.max(abs(differences))]

  c(ks_statistic = ks_statistic, signed = signed_ks_statistic)
}
